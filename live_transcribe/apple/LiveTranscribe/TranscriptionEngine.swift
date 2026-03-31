import Observation
import Speech
import Translation

enum RecognitionBackend: String, CaseIterable, Identifiable, Sendable {
    case speechAnalyzer = "SpeechAnalyzer"
    case sfSpeechRecognizer = "SFSpeechRecognizer"

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .speechAnalyzer: "SpeechAnalyzer (Offline)"
        case .sfSpeechRecognizer: "SFSpeechRecognizer (Online)"
        }
    }
}

@Observable
@MainActor
final class TranscriptionEngine {

    // MARK: - Published State

    var entries: [TranscriptEntry] = []
    var volatileText = ""
    var volatileTranslation = ""
    var isListening = false
    var sourceLanguage: AppLanguage = .korean
    var targetLanguage: AppLanguage = .english
    var isTranslationEnabled = true
    var backend: RecognitionBackend = .speechAnalyzer
    var status = "Ready"
    var errorMessage: String?

    // Audio level metrics for waveform visualization
    var audioRMS: Float = 0
    var audioGain: Float = 1
    var waveformSamples: [Float] = Array(repeating: 0, count: AudioMetrics.waveformLength)

    var translationConfig: TranslationSession.Configuration? {
        guard isTranslationEnabled, sourceLanguage != targetLanguage else { return nil }
        return .init(
            source: sourceLanguage.localeLanguage,
            target: targetLanguage.localeLanguage
        )
    }

    // MARK: - Private State

    private var analyzerSession: SpeechAnalyzerSession?
    private var recognizerSession: SpeechRecognitionSession?
    private var resultTask: Task<Void, Never>?
    private var metricsTask: Task<Void, Never>?
    private var translationContinuation: AsyncStream<TranslationRequest>.Continuation?
    private var volatileTranslationContinuation: AsyncStream<String>.Continuation?
    private var volatileDebounceTask: Task<Void, Never>?
    private var silenceTask: Task<Void, Never>?

    /// How long audio must stay silent before finalizing volatile text (seconds).
    private let silenceTimeout: Duration = .milliseconds(1500)
    /// RMS below this level counts as silence.
    private let silenceThreshold: Float = 0.005

    // MARK: - Lifecycle

    func start() async {
        guard !isListening else { return }

        errorMessage = nil

        // SFSpeechRecognizer requires authorization; SpeechAnalyzer does not
        if backend == .sfSpeechRecognizer {
            status = "Requesting authorization..."
            let authStatus = await SpeechRecognitionSession.requestAuthorization()
            guard authStatus == .authorized else {
                errorMessage = "Speech recognition not authorized. Check System Settings → Privacy."
                status = "Not authorized"
                return
            }
        }

        status = "Starting..."

        let resultStream: AsyncStream<SpeechAnalyzerSession.Result>

        switch backend {
        case .speechAnalyzer:
            let session = SpeechAnalyzerSession()
            self.analyzerSession = session
            do {
                // SpeechAnalyzerSession.Result and SpeechRecognitionSession.Result
                // have the same shape — we use SpeechAnalyzerSession.Result as the common type
                resultStream = try session.start(locale: sourceLanguage.locale)
            } catch {
                errorMessage = error.localizedDescription
                status = "Error"
                analyzerSession = nil
                return
            }

        case .sfSpeechRecognizer:
            let session = SpeechRecognitionSession()
            self.recognizerSession = session
            do {
                let sfStream = try session.start(locale: sourceLanguage.locale)
                resultStream = eraseToAsyncStream(
                    sfStream.map {
                        SpeechAnalyzerSession.Result(
                            text: $0.text, isFinal: $0.isFinal, error: $0.error
                        )
                    }
                )
            } catch {
                errorMessage = error.localizedDescription
                status = "Error"
                recognizerSession = nil
                return
            }
        }

        isListening = true
        status = "Listening..."

        // Poll audio metrics ~20 times per second for smooth waveform
        // and detect silence to finalize lingering volatile text
        let metrics: AudioMetrics? = analyzerSession?.audioMetrics ?? recognizerSession?.audioMetrics
        metricsTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(50))
                guard let self, let metrics else { break }
                self.audioRMS = metrics.rms
                self.audioGain = metrics.gain
                self.waveformSamples = metrics.waveform

                // Silence detection: if we have volatile text and audio is silent,
                // start a timer to finalize it
                if !self.volatileText.isEmpty && self.audioRMS < self.silenceThreshold {
                    if self.silenceTask == nil {
                        self.silenceTask = Task { [weak self] in
                            try? await Task.sleep(for: self?.silenceTimeout ?? .milliseconds(1500))
                            guard !Task.isCancelled, let self else { return }
                            // Still silent and still have volatile text? Finalize it.
                            if !self.volatileText.isEmpty && self.audioRMS < self.silenceThreshold {
                                self.finalizeVolatileText()
                            }
                            self.silenceTask = nil
                        }
                    }
                } else {
                    // Audio is active or no volatile text — cancel silence timer
                    self.silenceTask?.cancel()
                    self.silenceTask = nil
                }
            }
        }

        // Consume results on MainActor
        resultTask = Task { [weak self] in
            for await result in resultStream {
                guard let self, self.isListening else { break }

                if let error = result.error {
                    self.errorMessage = "Recognition error: \(error)"
                    continue
                }

                if result.isFinal {
                    var entry = TranscriptEntry(
                        text: result.text, language: self.sourceLanguage
                    )
                    let shouldTranslate =
                        self.isTranslationEnabled
                        && self.sourceLanguage != self.targetLanguage
                    if shouldTranslate {
                        entry.isTranslating = true
                    }

                    self.entries.append(entry)
                    self.volatileText = ""
                    self.volatileTranslation = ""
                    self.volatileDebounceTask?.cancel()

                    if shouldTranslate {
                        self.translationContinuation?.yield(
                            TranslationRequest(id: entry.id, text: result.text)
                        )
                    }
                } else {
                    self.volatileText = result.text
                    self.scheduleVolatileTranslation(result.text)
                }
            }
        }
    }

    func stop() async {
        analyzerSession?.stop()
        recognizerSession?.stop()
        resultTask?.cancel()
        metricsTask?.cancel()
        volatileDebounceTask?.cancel()
        silenceTask?.cancel()

        analyzerSession = nil
        recognizerSession = nil
        resultTask = nil
        metricsTask = nil
        volatileDebounceTask = nil
        silenceTask = nil
        isListening = false
        volatileText = ""
        volatileTranslation = ""
        audioRMS = 0
        audioGain = 1
        waveformSamples = Array(repeating: 0, count: AudioMetrics.waveformLength)
        status = "Stopped"
    }

    // MARK: - Translation helpers

    func makeTranslationStream() -> (
        stream: AsyncStream<TranslationRequest>,
        pending: [(id: UUID, text: String)]
    ) {
        translationContinuation?.finish()

        let (stream, continuation) = AsyncStream<TranslationRequest>.makeStream()
        translationContinuation = continuation

        let pending: [(id: UUID, text: String)] = entries
            .filter { $0.isTranslating && $0.translation == nil }
            .map { (id: $0.id, text: $0.text) }

        return (stream, pending)
    }

    func completeTranslation(id: UUID, result: String?) {
        guard let idx = entries.firstIndex(where: { $0.id == id }) else { return }
        entries[idx].translation = result
        entries[idx].isTranslating = false
    }

    // MARK: - Volatile (real-time) translation

    func makeVolatileTranslationStream() -> AsyncStream<String> {
        volatileTranslationContinuation?.finish()

        let (stream, continuation) = AsyncStream<String>.makeStream()
        volatileTranslationContinuation = continuation
        return stream
    }

    func completeVolatileTranslation(_ text: String?) {
        volatileTranslation = text ?? ""
    }

    private func scheduleVolatileTranslation(_ text: String) {
        volatileDebounceTask?.cancel()
        let shouldTranslate = isTranslationEnabled && sourceLanguage != targetLanguage
        guard shouldTranslate, !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return
        }

        volatileDebounceTask = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(500))
            guard !Task.isCancelled, let self else { return }
            self.volatileTranslationContinuation?.yield(text)
        }
    }

    // MARK: - Silence-based finalization

    /// Promotes the current volatile text to a finalized entry when silence is detected.
    private func finalizeVolatileText() {
        let text = volatileText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }

        var entry = TranscriptEntry(text: text, language: sourceLanguage)
        let shouldTranslate = isTranslationEnabled && sourceLanguage != targetLanguage
        if shouldTranslate {
            entry.isTranslating = true
        }

        entries.append(entry)
        volatileText = ""
        volatileTranslation = ""
        volatileDebounceTask?.cancel()

        if shouldTranslate {
            translationContinuation?.yield(
                TranslationRequest(id: entry.id, text: text)
            )
        }
    }

    // MARK: - Utilities

    func clearTranscript() {
        entries.removeAll()
        volatileText = ""
    }

    func exportTranscript() -> String {
        var output = "Transcript — \(Date().formatted())\n"
        output += String(repeating: "=", count: 60) + "\n\n"

        for entry in entries {
            let time = entry.timestamp.formatted(date: .omitted, time: .standard)
            output += "[\(time)] [\(entry.language.displayName)]\n"
            output += "  \(entry.text)\n"
            if let translation = entry.translation {
                output += "  → \(translation)\n"
            }
            output += "\n"
        }
        return output
    }
}

// MARK: - AsyncStream helpers

private func eraseToAsyncStream<S: AsyncSequence & Sendable>(
    _ sequence: S
) -> AsyncStream<S.Element> where S.Element: Sendable {
    AsyncStream { continuation in
        Task.detached {
            do {
                for try await element in sequence {
                    continuation.yield(element)
                }
            } catch {}
            continuation.finish()
        }
    }
}
