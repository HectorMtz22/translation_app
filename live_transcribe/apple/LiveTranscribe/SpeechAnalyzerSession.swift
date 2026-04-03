@preconcurrency import AVFoundation
import OSLog
import Speech

private let logger = Logger(subsystem: "com.livetranscribe", category: "SpeechAnalyzerSession")

/// Wraps SpeechAnalyzer + AVAudioEngine in a fully nonisolated class so that
/// no closures inherit @MainActor isolation (avoiding Swift 6 dispatch_assert_queue traps).
///
/// Results are delivered via an AsyncStream of simple Sendable values.
/// The analyzer is automatically restarted every `maxFinalResultsBeforeRestart`
/// final results to prevent recognition quality degradation over long sessions.
final class SpeechAnalyzerSession: @unchecked Sendable {

    struct Result: Sendable {
        let text: String
        let isFinal: Bool
        let error: String?
    }

    private var analyzer: SpeechAnalyzer?
    private var transcriber: DictationTranscriber?
    private var audioEngine: AVAudioEngine?
    private var inputContinuation: AsyncStream<AnalyzerInput>.Continuation?
    private var isStopped = false
    let audioMetrics = AudioMetrics()

    /// How many final results before the analyzer is restarted to keep quality high.
    private let maxFinalResultsBeforeRestart = 15

    // MARK: - Custom Language Model

    /// Directory for storing compiled custom language model files.
    private static var customModelDirectory: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("CustomLanguageModels", isDirectory: true)
    }

    /// Cached Korean language model configuration, shared across all sessions.
    nonisolated(unsafe) private static var cachedKoreanModelConfig: SFSpeechLanguageModel.Configuration?

    /// Version tag — bump this when the phrase list changes to invalidate the cache.
    private static let koreanModelVersion = "1.0"

    /// Prepares a custom Korean language model with common phrases to boost recognition accuracy.
    /// Returns a cached configuration if the compiled model already exists on disk.
    private static func prepareKoreanLanguageModel() async throws -> SFSpeechLanguageModel.Configuration {
        // Fast path: return in-memory cached config
        if let cached = cachedKoreanModelConfig {
            logger.info("Using cached Korean language model (in-memory)")
            return cached
        }

        let modelDir = customModelDirectory
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)

        let trainingDataURL = modelDir.appendingPathComponent("ko_training_data.bin")
        let compiledModelURL = modelDir.appendingPathComponent("ko_language_model")
        let compiledVocabURL = modelDir.appendingPathComponent("ko_vocabulary")
        let versionFile = modelDir.appendingPathComponent("ko_model_version.txt")

        let config = SFSpeechLanguageModel.Configuration(
            languageModel: compiledModelURL,
            vocabulary: compiledVocabURL
        )

        // Check if compiled model already exists on disk with matching version
        let fm = FileManager.default
        let currentVersion = (try? String(contentsOf: versionFile, encoding: .utf8)) ?? ""
        if currentVersion == koreanModelVersion
            && fm.fileExists(atPath: compiledModelURL.path)
            && fm.fileExists(atPath: compiledVocabURL.path) {
            logger.info("Using cached Korean language model (on-disk)")
            cachedKoreanModelConfig = config
            return config
        }

        // Build training data with common Korean conversational phrases
        let customData = SFCustomLanguageModelData(
            locale: Locale(identifier: "ko-KR"),
            identifier: "com.livetranscribe.korean",
            version: koreanModelVersion
        ) {
            // Common conversational phrases
            SFCustomLanguageModelData.PhraseCount(phrase: "안녕하세요", count: 100)
            SFCustomLanguageModelData.PhraseCount(phrase: "감사합니다", count: 100)
            SFCustomLanguageModelData.PhraseCount(phrase: "네 알겠습니다", count: 80)
            SFCustomLanguageModelData.PhraseCount(phrase: "그래서 이제 어떻게 할까요", count: 60)
            SFCustomLanguageModelData.PhraseCount(phrase: "잠깐만요", count: 60)
            SFCustomLanguageModelData.PhraseCount(phrase: "다시 한번 말씀해 주세요", count: 60)
            SFCustomLanguageModelData.PhraseCount(phrase: "좋습니다 진행하겠습니다", count: 50)
            SFCustomLanguageModelData.PhraseCount(phrase: "그렇구나", count: 50)
            SFCustomLanguageModelData.PhraseCount(phrase: "맞습니다", count: 50)
            SFCustomLanguageModelData.PhraseCount(phrase: "아 그렇군요", count: 50)
            SFCustomLanguageModelData.PhraseCount(phrase: "무슨 말씀이세요", count: 40)
            SFCustomLanguageModelData.PhraseCount(phrase: "이해했습니다", count: 40)
            SFCustomLanguageModelData.PhraseCount(phrase: "죄송합니다", count: 40)
            SFCustomLanguageModelData.PhraseCount(phrase: "괜찮습니다", count: 40)
            SFCustomLanguageModelData.PhraseCount(phrase: "어떻게 생각하세요", count: 30)
            SFCustomLanguageModelData.PhraseCount(phrase: "그러면 이렇게 하죠", count: 30)
            SFCustomLanguageModelData.PhraseCount(phrase: "확인해 보겠습니다", count: 30)
            SFCustomLanguageModelData.PhraseCount(phrase: "말씀하신 대로", count: 30)
            SFCustomLanguageModelData.PhraseCount(phrase: "잘 모르겠는데요", count: 30)
            SFCustomLanguageModelData.PhraseCount(phrase: "그건 좀 어려울 것 같아요", count: 20)
        }

        try await customData.export(to: trainingDataURL)
        logger.info("Exported Korean custom language model training data")

        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            SFSpeechLanguageModel.prepareCustomLanguageModel(
                for: trainingDataURL,
                configuration: config
            ) { error in
                if let error {
                    continuation.resume(throwing: error)
                } else {
                    continuation.resume()
                }
            }
        }

        // Write version marker so we can skip recompilation next launch
        try? koreanModelVersion.write(to: versionFile, atomically: true, encoding: .utf8)

        logger.info("Korean custom language model compiled successfully")
        cachedKoreanModelConfig = config
        return config
    }

    // MARK: - Start / Stop

    /// Starts speech recognition via SpeechAnalyzer with DictationTranscriber and returns an AsyncStream of results.
    /// The analyzer automatically restarts periodically to maintain recognition quality.
    func start(locale: Locale) throws -> AsyncStream<Result> {
        let (resultStream, resultContinuation) = AsyncStream<Result>.makeStream(
            bufferingPolicy: .bufferingNewest(20)
        )
        let capturedLocale = locale

        // Start a detached task to handle the async setup and result consumption
        Task.detached { [weak self] in
            do {
                // Prepare custom language model for Korean (once)
                var contentHints: Set<DictationTranscriber.ContentHint> = []
                let isKorean = capturedLocale.identifier.hasPrefix("ko")
                if isKorean {
                    do {
                        logger.info("Preparing custom Korean language model...")
                        let modelConfig = try await Self.prepareKoreanLanguageModel()
                        contentHints.insert(.customizedLanguage(modelConfiguration: modelConfig))
                        logger.info("Custom Korean language model ready")
                    } catch {
                        logger.warning("Failed to prepare Korean language model, continuing without it: \(error.localizedDescription)")
                        // Surface the error so the user knows recognition may be degraded
                        resultContinuation.yield(
                            Result(text: "", isFinal: false, error: "Korean language model unavailable — recognition may be less accurate")
                        )
                    }
                }

                // Query the optimal audio format using a temporary transcriber
                let formatQueryTranscriber = DictationTranscriber(
                    locale: capturedLocale,
                    contentHints: contentHints,
                    transcriptionOptions: [.punctuation],
                    reportingOptions: [.volatileResults],
                    attributeOptions: [.audioTimeRange]
                )
                let analyzerFormat = await SpeechAnalyzer.bestAvailableAudioFormat(
                    compatibleWith: [formatQueryTranscriber]
                )

                // Set up audio engine once — it stays running across analyzer restarts
                try self?.setupAudioEngine(analyzerFormat: analyzerFormat)

                // Restart loop — recreates the analyzer periodically to prevent degradation
                while let self, !self.isStopped, !Task.isCancelled {
                    let finalCount = try await self.runAnalyzerCycle(
                        locale: capturedLocale,
                        contentHints: contentHints,
                        resultContinuation: resultContinuation
                    )

                    if self.isStopped || Task.isCancelled { break }

                    logger.info("Restarting analyzer after \(finalCount) final results to maintain quality")
                }

                resultContinuation.finish()
            } catch {
                if !Task.isCancelled {
                    logger.error("SpeechAnalyzerSession error: \(error.localizedDescription)")
                    resultContinuation.yield(
                        Result(text: "", isFinal: false, error: error.localizedDescription)
                    )
                }
                resultContinuation.finish()
            }
        }

        return resultStream
    }

    /// Runs a single analyzer cycle. Returns the number of final results received
    /// before the cycle ends (either due to reaching the restart threshold or the stream ending).
    private func runAnalyzerCycle(
        locale: Locale,
        contentHints: Set<DictationTranscriber.ContentHint>,
        resultContinuation: AsyncStream<Result>.Continuation
    ) async throws -> Int {
        let transcriber = DictationTranscriber(
            locale: locale,
            contentHints: contentHints,
            transcriptionOptions: [.punctuation],
            reportingOptions: [.volatileResults],
            attributeOptions: [.audioTimeRange]
        )
        self.transcriber = transcriber

        let analyzer = SpeechAnalyzer(modules: [transcriber])
        self.analyzer = analyzer

        // Download model if needed (no-op after first download)
        if let downloader = try await AssetInventory.assetInstallationRequest(
            supporting: [transcriber]
        ) {
            logger.info("Downloading speech model for \(locale.identifier)...")
            do {
                try await downloader.downloadAndInstall()
                logger.info("Speech model download complete for \(locale.identifier)")
            } catch {
                logger.error("Speech model download failed for \(locale.identifier): \(error.localizedDescription)")
                resultContinuation.yield(
                    Result(text: "", isFinal: false, error: "Speech model download failed: \(error.localizedDescription)")
                )
                throw error
            }
        }

        // Create input stream for this analyzer cycle
        let (inputStream, inputCont) = AsyncStream<AnalyzerInput>.makeStream()
        self.inputContinuation = inputCont

        // Start the analyzer
        try await analyzer.start(inputSequence: inputStream)
        logger.info("Analyzer cycle started for \(locale.identifier)")

        // Consume transcription results until we hit the restart threshold
        var finalResultCount = 0
        for try await result in transcriber.results {
            guard !isStopped, !Task.isCancelled else { break }

            let fullText = String(result.text.characters)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard !fullText.isEmpty else { continue }

            resultContinuation.yield(
                Result(text: fullText, isFinal: result.isFinal, error: nil)
            )

            if result.isFinal {
                finalResultCount += 1
                if finalResultCount >= maxFinalResultsBeforeRestart {
                    break
                }
            }
        }

        // Create a temporary "bridge" continuation so the audio tap keeps flowing
        // while we tear down the old analyzer. The next cycle will replace it.
        let (bridgeStream, bridgeCont) = AsyncStream<AnalyzerInput>.makeStream()
        self.inputContinuation = bridgeCont
        // Discard bridge buffers (they're just to avoid dropping audio during teardown)
        Task.detached { for await _ in bridgeStream {} }

        // Now tear down this cycle's analyzer
        inputCont.finish()
        try? await analyzer.finalizeAndFinishThroughEndOfInput()
        self.analyzer = nil
        self.transcriber = nil

        return finalResultCount
    }

    private func setupAudioEngine(analyzerFormat: AVAudioFormat?) throws {
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
            throw SessionError.noInput
        }

        // SpeechAnalyzer wants 1ch 16kHz Int16 — AVAudioEngine can't convert
        // to Int16 natively in installTap, so we use a manual converter.
        let converter: AVAudioConverter?
        if let analyzerFormat {
            converter = AVAudioConverter(from: micFormat, to: analyzerFormat)
        } else {
            converter = nil
        }

        let targetFormat = analyzerFormat

        // Serial queue for format conversion (off the realtime audio thread)
        let convertQueue = DispatchQueue(
            label: "com.livetranscribe.audio-convert",
            qos: .userInteractive
        )

        // The tap reads self.inputContinuation each time, so when we restart the
        // analyzer and swap the continuation, audio flows to the new analyzer.
        let metrics = self.audioMetrics
        inputNode.installTap(onBus: 0, bufferSize: 4096, format: micFormat) { [weak self] buffer, _ in
            // Amplify audio for better sensitivity to quiet voices
            Self.adaptiveAmplify(buffer, metrics: metrics)

            convertQueue.async { [weak self] in
                guard let currentContinuation = self?.inputContinuation else { return }

                guard let targetFormat, let converter else {
                    currentContinuation.yield(AnalyzerInput(buffer: buffer))
                    return
                }

                let ratio = targetFormat.sampleRate / buffer.format.sampleRate
                let frameCount = AVAudioFrameCount(Double(buffer.frameLength) * ratio)
                guard frameCount > 0,
                      let output = AVAudioPCMBuffer(
                          pcmFormat: targetFormat, frameCapacity: frameCount
                      )
                else { return }

                var error: NSError?
                var consumed = false
                let status = converter.convert(to: output, error: &error) { _, outStatus in
                    if consumed {
                        outStatus.pointee = .noDataNow
                        return nil
                    }
                    consumed = true
                    outStatus.pointee = .haveData
                    return buffer
                }

                if status == .haveData || status == .inputRanDry {
                    currentContinuation.yield(AnalyzerInput(buffer: output))
                }
            }
        }

        engine.prepare()
        try engine.start()
        self.audioEngine = engine
    }

    func stop() {
        isStopped = true

        audioEngine?.stop()
        audioEngine?.inputNode.removeTap(onBus: 0)
        inputContinuation?.finish()

        Task {
            try? await analyzer?.finalizeAndFinishThroughEndOfInput()
        }

        audioEngine = nil
        analyzer = nil
        transcriber = nil
        inputContinuation = nil
    }

    // MARK: - Adaptive Audio Amplification

    /// Adaptively amplifies quiet audio buffers to improve speech recognition sensitivity.
    /// Only boosts audio when the level is below a threshold — leaves normal/loud audio untouched.
    /// Updates the provided AudioMetrics with the current RMS level and applied gain.
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
        case noInput

        var errorDescription: String? {
            "No audio input device. Check System Settings → Sound."
        }
    }
}
