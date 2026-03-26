import Foundation

// MARK: - Supported Languages

enum AppLanguage: String, CaseIterable, Identifiable, Codable, Sendable {
    case korean = "ko"
    case english = "en"
    case spanish = "es"

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .korean: "Korean"
        case .english: "English"
        case .spanish: "Spanish"
        }
    }

    var locale: Locale {
        switch self {
        case .korean: Locale(identifier: "ko-KR")
        case .english: Locale(identifier: "en-US")
        case .spanish: Locale(identifier: "es-ES")
        }
    }

    var localeLanguage: Locale.Language {
        Locale.Language(identifier: rawValue)
    }
}

// MARK: - Transcript Entry

struct TranscriptEntry: Identifiable {
    let id: UUID
    let text: String
    let language: AppLanguage
    let timestamp: Date
    var translation: String?
    var isTranslating: Bool

    init(text: String, language: AppLanguage, timestamp: Date = Date()) {
        self.id = UUID()
        self.text = text
        self.language = language
        self.timestamp = timestamp
        self.translation = nil
        self.isTranslating = false
    }
}

// MARK: - Translation Request

struct TranslationRequest: Sendable {
    let id: UUID
    let text: String
}

// MARK: - Audio Metrics

/// Thread-safe container for real-time audio level data, updated from the audio tap
/// and read by the UI for waveform visualization.
final class AudioMetrics: @unchecked Sendable {
    /// Number of RMS samples to keep for the waveform display.
    static let waveformLength = 40

    private let lock = NSLock()
    private var _rms: Float = 0
    private var _gain: Float = 1
    private var _waveform: [Float] = Array(repeating: 0, count: waveformLength)

    var rms: Float {
        lock.withLock { _rms }
    }

    var gain: Float {
        lock.withLock { _gain }
    }

    var waveform: [Float] {
        lock.withLock { _waveform }
    }

    func update(rms: Float, gain: Float) {
        lock.withLock {
            _rms = rms
            _gain = gain
            _waveform.append(rms)
            if _waveform.count > Self.waveformLength {
                _waveform.removeFirst(_waveform.count - Self.waveformLength)
            }
        }
    }

    func reset() {
        lock.withLock {
            _rms = 0
            _gain = 1
            _waveform = Array(repeating: 0, count: Self.waveformLength)
        }
    }
}
