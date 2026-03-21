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
