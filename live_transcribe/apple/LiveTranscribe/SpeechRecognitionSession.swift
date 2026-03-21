@preconcurrency import AVFoundation
import Speech

/// Wraps SFSpeechRecognizer + AVAudioEngine in a nonisolated class so that
/// none of the framework callbacks inherit @MainActor isolation (which would
/// trigger Swift 6 dispatch_assert_queue traps at runtime).
///
/// Results are delivered via an AsyncStream of simple Sendable values.
final class SpeechRecognitionSession: @unchecked Sendable {

    struct Result: Sendable {
        let text: String
        let isFinal: Bool
        let error: String?
    }

    private var recognizer: SFSpeechRecognizer?
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private var audioEngine: AVAudioEngine?
    private var continuation: AsyncStream<Result>.Continuation?

    // MARK: - Authorization

    static func requestAuthorization() async -> SFSpeechRecognizerAuthorizationStatus {
        await withCheckedContinuation { continuation in
            SFSpeechRecognizer.requestAuthorization { status in
                continuation.resume(returning: status)
            }
        }
    }

    // MARK: - Start / Stop

    /// Starts speech recognition and returns an AsyncStream of results.
    /// The stream ends when `stop()` is called or an unrecoverable error occurs.
    func start(locale: Locale) throws -> AsyncStream<Result> {
        guard let recognizer = SFSpeechRecognizer(locale: locale) else {
            throw SessionError.unavailable("Speech recognition not available for this language.")
        }
        guard recognizer.isAvailable else {
            throw SessionError.unavailable(
                "Speech recognizer unavailable. Enable Dictation in System Settings → Keyboard → Dictation."
            )
        }
        self.recognizer = recognizer

        #if os(iOS)
        let audioSession = AVAudioSession.sharedInstance()
        try audioSession.setCategory(.record, mode: .measurement)
        try audioSession.setActive(true)
        #endif

        let engine = AVAudioEngine()
        let inputNode = engine.inputNode
        let micFormat = inputNode.outputFormat(forBus: 0)

        guard micFormat.sampleRate > 0, micFormat.channelCount > 0 else {
            throw SessionError.noInput("No audio input device. Check System Settings → Sound.")
        }

        let (stream, continuation) = AsyncStream<Result>.makeStream()
        self.continuation = continuation

        // Create recognition request
        let request = SFSpeechAudioBufferRecognitionRequest()
        if recognizer.supportsOnDeviceRecognition {
            request.requiresOnDeviceRecognition = true
        }
        request.shouldReportPartialResults = true
        request.addsPunctuation = true
        self.request = request

        // Install audio tap — feeds raw audio to the recognition request
        inputNode.installTap(onBus: 0, bufferSize: 4096, format: micFormat) { buffer, _ in
            request.append(buffer)
        }

        engine.prepare()
        try engine.start()
        self.audioEngine = engine

        // Start recognition
        startRecognitionTask(recognizer: recognizer, request: request)

        return stream
    }

    func stop() {
        request?.endAudio()
        recognitionTask?.cancel()

        audioEngine?.stop()
        audioEngine?.inputNode.removeTap(onBus: 0)

        continuation?.finish()

        audioEngine = nil
        recognizer = nil
        request = nil
        recognitionTask = nil
        continuation = nil
    }

    // MARK: - Internal

    private var lastFinalizedText = ""

    private func startRecognitionTask(
        recognizer: SFSpeechRecognizer,
        request: SFSpeechAudioBufferRecognitionRequest
    ) {
        recognitionTask = recognizer.recognitionTask(with: request) { [weak self] result, error in
            guard let self else { return }

            if let error {
                let nsError = error as NSError
                // Ignore known transient errors:
                // 203 = "Retry" (recognizer wants to restart)
                // 209 = recognition cancelled
                // 216 = request superseded
                // 301 = audio engine error (transient)
                let ignoreCodes: Set<Int> = [203, 209, 216, 301]
                if nsError.domain == "kAFAssistantErrorDomain"
                    && ignoreCodes.contains(nsError.code)
                {
                    return
                }
                self.continuation?.yield(Result(text: "", isFinal: false, error: error.localizedDescription))
                return
            }

            guard let result else { return }
            let text = result.bestTranscription.formattedString

            if result.isFinal {
                guard text != self.lastFinalizedText, !text.isEmpty else { return }
                self.lastFinalizedText = text
                self.continuation?.yield(Result(text: text, isFinal: true, error: nil))

                // SFSpeechRecognizer sends one final result per task — restart
                self.restartRecognition()
            } else {
                // Partial result — strip already-finalized prefix
                let partial = text.hasPrefix(self.lastFinalizedText)
                    ? String(text.dropFirst(self.lastFinalizedText.count))
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                    : text
                if !partial.isEmpty {
                    self.continuation?.yield(Result(text: partial, isFinal: false, error: nil))
                }
            }
        }
    }

    private func restartRecognition() {
        guard let recognizer, recognizer.isAvailable else { return }

        recognitionTask?.cancel()
        recognitionTask = nil
        request = nil

        let newRequest = SFSpeechAudioBufferRecognitionRequest()
        if recognizer.supportsOnDeviceRecognition {
            newRequest.requiresOnDeviceRecognition = true
        }
        newRequest.shouldReportPartialResults = true
        newRequest.addsPunctuation = true
        self.request = newRequest

        // Reinstall tap for the new request
        audioEngine?.inputNode.removeTap(onBus: 0)
        if let inputNode = audioEngine?.inputNode {
            let micFormat = inputNode.outputFormat(forBus: 0)
            inputNode.installTap(onBus: 0, bufferSize: 4096, format: micFormat) { buffer, _ in
                newRequest.append(buffer)
            }
        }

        lastFinalizedText = ""
        startRecognitionTask(recognizer: recognizer, request: newRequest)
    }

    // MARK: - Errors

    enum SessionError: LocalizedError {
        case unavailable(String)
        case noInput(String)

        var errorDescription: String? {
            switch self {
            case .unavailable(let msg), .noInput(let msg): msg
            }
        }
    }
}
