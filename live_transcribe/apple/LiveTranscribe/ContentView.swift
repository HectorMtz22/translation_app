import SwiftUI
@preconcurrency import Translation

struct ContentView: View {
    @State private var engine = TranscriptionEngine()

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                controlsBar
                Divider()
                transcriptList
            }
            .navigationTitle("Live Transcribe")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItemGroup {
                    if !engine.entries.isEmpty {
                        ShareLink(
                            item: engine.exportTranscript(),
                            preview: SharePreview("Transcript")
                        ) {
                            Label("Export", systemImage: "square.and.arrow.up")
                        }

                        Button(role: .destructive) {
                            engine.clearTranscript()
                        } label: {
                            Label("Clear", systemImage: "trash")
                        }
                    }
                }
            }
            // Translation session — stays alive while translationConfig is non-nil.
            // The session is used here (not sent to the engine) to avoid
            // Swift 6 actor-isolation issues.
            .translationTask(engine.translationConfig) { session in
                // Get stream + pending entries from engine
                let (stream, pending) = engine.makeTranslationStream()

                // Batch-translate already-pending entries
                for item in pending {
                    let response = try? await session.translate(item.text)
                    engine.completeTranslation(
                        id: item.id, result: response?.targetText
                    )
                }

                // Translate new entries as they arrive
                for await request in stream {
                    guard !Task.isCancelled else { break }
                    let response = try? await session.translate(request.text)
                    engine.completeTranslation(
                        id: request.id, result: response?.targetText
                    )
                }
            }
        }
    }

    // MARK: - Controls

    @ViewBuilder
    private var controlsBar: some View {
        VStack(spacing: 12) {
            HStack {
                Picker("Engine", selection: $engine.backend) {
                    ForEach(RecognitionBackend.allCases) { b in
                        Text(b.displayName).tag(b)
                    }
                }
                .disabled(engine.isListening)

                Picker("From", selection: $engine.sourceLanguage) {
                    ForEach(AppLanguage.allCases) { lang in
                        Text(lang.displayName).tag(lang)
                    }
                }
                .disabled(engine.isListening)

                Toggle("Translate", isOn: $engine.isTranslationEnabled)
                    .fixedSize()

                if engine.isTranslationEnabled {
                    Picker("To", selection: $engine.targetLanguage) {
                        ForEach(
                            AppLanguage.allCases.filter { $0 != engine.sourceLanguage }
                        ) { lang in
                            Text(lang.displayName).tag(lang)
                        }
                    }
                }
            }

            HStack {
                Button {
                    Task {
                        if engine.isListening {
                            await engine.stop()
                        } else {
                            await engine.start()
                        }
                    }
                } label: {
                    Label(
                        engine.isListening ? "Stop" : "Start",
                        systemImage: engine.isListening
                            ? "stop.circle.fill"
                            : "mic.circle.fill"
                    )
                    .font(.headline)
                }
                .buttonStyle(.borderedProminent)
                .tint(engine.isListening ? .red : .accentColor)

                Spacer()

                Text(engine.status)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            if let error = engine.errorMessage {
                Text(error)
                    .font(.caption)
                    .foregroundStyle(.red)
            }
        }
        .padding()
    }

    // MARK: - Transcript List

    @ViewBuilder
    private var transcriptList: some View {
        ScrollViewReader { proxy in
            List {
                ForEach(engine.entries) { entry in
                    TranscriptRow(entry: entry)
                        .id(entry.id)
                }

                if !engine.volatileText.isEmpty {
                    Text(engine.volatileText)
                        .foregroundStyle(.secondary)
                        .italic()
                        .id("volatile")
                }
            }
            .listStyle(.plain)
            .onChange(of: engine.entries.count) {
                withAnimation {
                    if let last = engine.entries.last {
                        proxy.scrollTo(last.id, anchor: .bottom)
                    }
                }
            }
            .onChange(of: engine.volatileText) {
                proxy.scrollTo("volatile", anchor: .bottom)
            }
            .overlay {
                if engine.entries.isEmpty && !engine.isListening {
                    ContentUnavailableView(
                        "No Transcript",
                        systemImage: "waveform",
                        description: Text("Tap Start to begin transcribing.")
                    )
                }
            }
        }
    }
}

// MARK: - Transcript Row

private struct TranscriptRow: View {
    let entry: TranscriptEntry

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                Text(entry.timestamp, style: .time)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)

                Text(entry.language.displayName)
                    .font(.caption2)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(.quaternary, in: .capsule)
            }

            Text(entry.text)
                .font(.body)
                .textSelection(.enabled)

            if entry.isTranslating {
                ProgressView()
                    .controlSize(.small)
            } else if let translation = entry.translation {
                Text("→ \(translation)")
                    .font(.body)
                    .foregroundStyle(.orange)
                    .textSelection(.enabled)
            }
        }
        .padding(.vertical, 4)
    }
}

#Preview {
    ContentView()
}
