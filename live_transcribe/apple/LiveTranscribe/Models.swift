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
///
/// Uses a fixed-size circular buffer for O(1) append instead of Array's O(n) removeFirst.
final class AudioMetrics: @unchecked Sendable {
    /// Number of RMS samples to keep for the waveform display.
    static let waveformLength = 40

    private let lock = NSLock()
    private var _rms: Float = 0
    private var _gain: Float = 1
    /// Circular buffer backing storage.
    private var _ring = [Float](repeating: 0, count: waveformLength)
    /// Write index into _ring (wraps around).
    private var _ringIndex = 0

    var rms: Float {
        lock.withLock { _rms }
    }

    var gain: Float {
        lock.withLock { _gain }
    }

    /// Returns the waveform samples in chronological order.
    var waveform: [Float] {
        lock.withLock {
            let len = Self.waveformLength
            // _ringIndex points to the next write slot, so the oldest sample is at _ringIndex
            let start = _ringIndex % len
            return Array(_ring[start..<len]) + Array(_ring[0..<start])
        }
    }

    func update(rms: Float, gain: Float) {
        lock.withLock {
            _rms = rms
            _gain = gain
            _ring[_ringIndex % Self.waveformLength] = rms
            _ringIndex = (_ringIndex + 1) % Self.waveformLength
        }
    }

    func reset() {
        lock.withLock {
            _rms = 0
            _gain = 1
            _ring = [Float](repeating: 0, count: Self.waveformLength)
            _ringIndex = 0
        }
    }
}
