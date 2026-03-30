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
    let audioMetrics = AudioMetrics()

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
        // Boost hardware input gain for better sensitivity to quiet voices
        if audioSession.isInputGainSettable {
            try? audioSession.setInputGain(1.0)
        }
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
        request.requiresOnDeviceRecognition = false
        request.shouldReportPartialResults = true
        request.addsPunctuation = true
        self.request = request

        // Install audio tap — amplify audio for better sensitivity to quiet voices
        let metrics = self.audioMetrics
        inputNode.installTap(onBus: 0, bufferSize: 4096, format: micFormat) { buffer, _ in
            Self.adaptiveAmplify(buffer, metrics: metrics)
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
            let metrics = self.audioMetrics
            inputNode.installTap(onBus: 0, bufferSize: 4096, format: micFormat) { buffer, _ in
                Self.adaptiveAmplify(buffer, metrics: metrics)
                newRequest.append(buffer)
            }
        }

        lastFinalizedText = ""
        startRecognitionTask(recognizer: recognizer, request: newRequest)
    }

    // MARK: - Adaptive Audio Amplification

    /// Adaptively amplifies quiet audio buffers to improve speech recognition sensitivity.
    /// Only boosts audio when the level is below a threshold — leaves normal/loud audio untouched.
    private static func adaptiveAmplify(_ buffer: AVAudioPCMBuffer, metrics: AudioMetrics) {
        guard let floatData = buffer.floatChannelData else { return }
        let frameCount = Int(buffer.frameLength)
        guard frameCount > 0 else { return }

        // Calculate RMS (root mean square) level of the buffer
        var sumSquares: Float = 0
        let channelData = floatData[0]
        for frame in 0..<frameCount {
            let sample = channelData[frame]
            sumSquares += sample * sample
        }
        let rms = sqrtf(sumSquares / Float(frameCount))

        // Only amplify if audio is quiet (RMS below threshold)
        let quietThreshold: Float = 0.05
        if rms < quietThreshold, rms > 0.00005 {
            let maxGain: Float = 10.0
            let targetRMS: Float = 0.08
            let gain = min(maxGain, targetRMS / rms)

            for channel in 0..<Int(buffer.format.channelCount) {
                let data = floatData[channel]
                for frame in 0..<frameCount {
                    data[frame] = max(-1.0, min(1.0, data[frame] * gain))
                }
            }
            metrics.update(rms: rms, gain: gain)
        } else {
            metrics.update(rms: rms, gain: 1.0)
        }
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
