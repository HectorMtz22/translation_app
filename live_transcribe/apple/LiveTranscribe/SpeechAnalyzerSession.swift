@preconcurrency import AVFoundation
import OSLog
import Speech

private let logger = Logger(subsystem: "com.livetranscribe", category: "SpeechAnalyzerSession")

/// Wraps SpeechAnalyzer + AVAudioEngine in a fully nonisolated class so that
/// no closures inherit @MainActor isolation (avoiding Swift 6 dispatch_assert_queue traps).
///
/// Results are delivered via an AsyncStream of simple Sendable values.
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

    // MARK: - Custom Language Model

    /// Directory for storing compiled custom language model files.
    private static var customModelDirectory: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("CustomLanguageModels", isDirectory: true)
    }

    /// Prepares a custom Korean language model with common phrases to boost recognition accuracy.
    private static func prepareKoreanLanguageModel() async throws -> SFSpeechLanguageModel.Configuration {
        let modelDir = customModelDirectory
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)

        let trainingDataURL = modelDir.appendingPathComponent("ko_training_data.bin")
        let compiledModelURL = modelDir.appendingPathComponent("ko_language_model")
        let compiledVocabURL = modelDir.appendingPathComponent("ko_vocabulary")

        // Build training data with common Korean conversational phrases
        let customData = SFCustomLanguageModelData(
            locale: Locale(identifier: "ko-KR"),
            identifier: "com.livetranscribe.korean",
            version: "1.0"
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

        let config = SFSpeechLanguageModel.Configuration(
            languageModel: compiledModelURL,
            vocabulary: compiledVocabURL
        )

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

        logger.info("Korean custom language model compiled successfully")
        return config
    }

    // MARK: - Start / Stop

    /// Starts speech recognition via SpeechAnalyzer with DictationTranscriber and returns an AsyncStream of results.
    func start(locale: Locale) throws -> AsyncStream<Result> {
        let (resultStream, resultContinuation) = AsyncStream<Result>.makeStream()
        let capturedLocale = locale

        // Start a detached task to handle the async setup and result consumption
        Task.detached { [weak self] in
            do {
                // Prepare custom language model for Korean
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
                    }
                }

                let transcriber = DictationTranscriber(
                    locale: capturedLocale,
                    contentHints: contentHints,
                    transcriptionOptions: [.punctuation],
                    reportingOptions: [.volatileResults],
                    attributeOptions: [.audioTimeRange]
                )
                self?.transcriber = transcriber

                let analyzer = SpeechAnalyzer(modules: [transcriber])
                self?.analyzer = analyzer

                // Download model if needed — verify full model installation
                logger.info("Checking asset installation for locale: \(capturedLocale.identifier)")
                if let downloader = try await AssetInventory.assetInstallationRequest(
                    supporting: [transcriber]
                ) {
                    logger.info("Downloading speech model for \(capturedLocale.identifier)...")
                    try await downloader.downloadAndInstall()
                    logger.info("Speech model download complete for \(capturedLocale.identifier)")
                } else {
                    logger.info("Speech model already installed for \(capturedLocale.identifier)")
                }

                // Verify the model is installed after download attempt
                let installedLocales = await DictationTranscriber.installedLocales
                let isInstalled = installedLocales.contains { $0.identifier.hasPrefix(capturedLocale.language.languageCode?.identifier ?? "") }
                if isInstalled {
                    logger.info("Verified: speech model installed for \(capturedLocale.identifier)")
                } else {
                    logger.warning("Speech model for \(capturedLocale.identifier) may not be fully installed. Installed locales: \(installedLocales.map(\.identifier))")
                }

                // Get optimal audio format
                let analyzerFormat = await SpeechAnalyzer.bestAvailableAudioFormat(
                    compatibleWith: [transcriber]
                )

                // Create input stream for the analyzer
                let (inputStream, inputCont) = AsyncStream<AnalyzerInput>.makeStream()
                self?.inputContinuation = inputCont

                // Start the analyzer
                try await analyzer.start(inputSequence: inputStream)

                // Set up audio engine
                try self?.setupAudioEngine(analyzerFormat: analyzerFormat)

                // Consume transcription results
                for try await result in transcriber.results {
                    let text = String(result.text.characters)
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                    guard !text.isEmpty else { continue }

                    resultContinuation.yield(
                        Result(text: text, isFinal: result.isFinal, error: nil)
                    )
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

    private func setupAudioEngine(analyzerFormat: AVAudioFormat?) throws {
        #if os(iOS)
        let audioSession = AVAudioSession.sharedInstance()
        try audioSession.setCategory(.record, mode: .measurement)
        try audioSession.setActive(true)
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

        let continuation = inputContinuation
        let targetFormat = analyzerFormat

        // Serial queue for format conversion (off the realtime audio thread)
        let convertQueue = DispatchQueue(
            label: "com.livetranscribe.audio-convert",
            qos: .userInteractive
        )

        inputNode.installTap(onBus: 0, bufferSize: 4096, format: micFormat) { buffer, _ in
            convertQueue.async {
                guard let targetFormat, let converter else {
                    continuation?.yield(AnalyzerInput(buffer: buffer))
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
                    continuation?.yield(AnalyzerInput(buffer: output))
                }
            }
        }

        engine.prepare()
        try engine.start()
        self.audioEngine = engine
    }

    func stop() {
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

    // MARK: - Errors

    enum SessionError: LocalizedError {
        case noInput

        var errorDescription: String? {
            "No audio input device. Check System Settings → Sound."
        }
    }
}
