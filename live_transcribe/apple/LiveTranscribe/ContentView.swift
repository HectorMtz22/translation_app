import SwiftUI
@preconcurrency import Translation

struct ContentView: View {
    @State private var engine = TranscriptionEngine()
    @State private var shouldAutoScroll = true

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                compactHeader
                controlsBar
                if engine.isListening {
                    AudioLevelView(
                        waveform: engine.waveformSamples,
                        rms: engine.audioRMS,
                        gain: engine.audioGain
                    )
                    .frame(height: horizontalSizeClass == .compact ? 36 : 60)
                    .padding(.horizontal)
                }
                Divider()
                transcriptList
            }
            #if os(iOS)
            .toolbarVisibility(.hidden, for: .navigationBar)
            #endif
        }
            // Translation session — stays alive while translationConfig is non-nil.
            // The session is used here (not sent to the engine) to avoid
            // Swift 6 actor-isolation issues.
            .translationTask(engine.translationConfig) { session in
                // Get stream + pending entries from engine
                let (stream, pending) = engine.makeTranslationStream()
                let volatileStream = engine.makeVolatileTranslationStream()

                // Translate volatile (partial) text in real-time
                let volatileTask = Task {
                    for await text in volatileStream {
                        guard !Task.isCancelled else { break }
                        do {
                            let response = try await session.translate(text)
                            if engine.volatileText == text {
                                engine.completeVolatileTranslation(response.targetText)
                            }
                        } catch {
                            // Volatile translation failures are non-critical — just clear it
                            if engine.volatileText == text {
                                engine.completeVolatileTranslation(nil)
                            }
                        }
                    }
                }

                // Batch-translate already-pending entries
                for item in pending {
                    do {
                        let response = try await session.translate(item.text)
                        engine.completeTranslation(
                            id: item.id, result: response.targetText
                        )
                    } catch {
                        engine.completeTranslation(id: item.id, result: nil)
                        engine.errorMessage = "Translation failed: \(error.localizedDescription)"
                    }
                }

                // Translate new entries as they arrive
                for await request in stream {
                    guard !Task.isCancelled else { break }
                    do {
                        let response = try await session.translate(request.text)
                        engine.completeTranslation(
                            id: request.id, result: response.targetText
                        )
                    } catch {
                        engine.completeTranslation(id: request.id, result: nil)
                        engine.errorMessage = "Translation failed: \(error.localizedDescription)"
                    }
                }

                volatileTask.cancel()
            }
    }

    // MARK: - Controls

    @Environment(\.horizontalSizeClass) private var horizontalSizeClass

    @ViewBuilder
    private var compactHeader: some View {
        HStack {
            Text("Live Transcribe")
                .font(horizontalSizeClass == .compact ? .subheadline.bold() : .headline.bold())

            Spacer()

            if !engine.entries.isEmpty {
                ShareLink(
                    item: engine.exportTranscript(),
                    preview: SharePreview("Transcript")
                ) {
                    Image(systemName: "square.and.arrow.up")
                        .font(.subheadline)
                }
                .buttonStyle(.borderless)

                Button(role: .destructive) {
                    engine.clearTranscript()
                } label: {
                    Image(systemName: "trash")
                        .font(.subheadline)
                }
                .buttonStyle(.borderless)
            }
        }
        .padding(.horizontal, horizontalSizeClass == .compact ? 12 : 16)
        .padding(.vertical, 6)
    }

    private let compactColumns = [
        GridItem(.flexible()),
        GridItem(.flexible()),
    ]

    @ViewBuilder
    private var controlsBar: some View {
        if horizontalSizeClass == .compact {
            LazyVGrid(columns: compactColumns, spacing: 12) {
                // Row 1
                Picker("Engine", selection: $engine.backend) {
                    ForEach(RecognitionBackend.allCases) { b in
                        Text(b.displayName).tag(b)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .disabled(engine.isListening)

                Picker("From", selection: $engine.sourceLanguage) {
                    ForEach(AppLanguage.allCases) { lang in
                        Text(lang.displayName).tag(lang)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .trailing)
                .disabled(engine.isListening)

                // Row 2
                Toggle("Translate", isOn: $engine.isTranslationEnabled)
                    .fixedSize()
                    .frame(maxWidth: .infinity, alignment: .leading)

                if engine.isTranslationEnabled {
                    Picker("To", selection: $engine.targetLanguage) {
                        ForEach(
                            AppLanguage.allCases.filter {
                                $0 != engine.sourceLanguage
                            }
                        ) { lang in
                            Text(lang.displayName).tag(lang)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .trailing)
                } else {
                    Spacer()
                }

                // Row 3
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
                .frame(maxWidth: .infinity, alignment: .leading)

                Text(engine.status)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .trailing)
            }
            .padding()

            if let error = engine.errorMessage {
                Text(error)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .padding(.horizontal)
            }
        } else {
            VStack(spacing: 8) {
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
                                AppLanguage.allCases.filter {
                                    $0 != engine.sourceLanguage
                                }
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
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }

                if let error = engine.errorMessage {
                    Text(error)
                        .font(.caption2)
                        .foregroundStyle(.red)
                }
            }
            .padding(16)
        }
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
                    VStack(alignment: .leading, spacing: 2) {
                        Text(engine.volatileText)
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                            .italic()
                        if !engine.volatileTranslation.isEmpty {
                            Text("→ \(engine.volatileTranslation)")
                                .font(.footnote)
                                .foregroundStyle(.orange.opacity(0.7))
                                .italic()
                        }
                    }
                    .id("volatile")
                }

                Color.clear
                    .frame(height: 1)
                    .id("bottom")
                    .onScrollVisibilityChange(threshold: 0.0) { visible in
                        shouldAutoScroll = visible
                    }
            }
            .listStyle(.plain)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .onChange(of: engine.entries.count) {
                if shouldAutoScroll {
                    withAnimation {
                        proxy.scrollTo("bottom", anchor: .bottom)
                    }
                }
            }
            .onChange(of: engine.volatileText) {
                if shouldAutoScroll {
                    proxy.scrollTo("bottom", anchor: .bottom)
                }
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

    @Environment(\.horizontalSizeClass) private var horizontalSizeClass

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 4) {
                Text(entry.timestamp, style: .time)
                    .font(.system(size: 10))
                    .foregroundStyle(.tertiary)

                Text(entry.language.displayName)
                    .font(.system(size: 10))
                    .padding(.horizontal, 4)
                    .padding(.vertical, 1)
                    .background(.quaternary, in: .capsule)
            }

            Text(entry.text)
                .font(horizontalSizeClass == .compact ? .footnote : .body)
                .textSelection(.enabled)

            if entry.isTranslating {
                ProgressView()
                    .controlSize(.mini)
            } else if let translation = entry.translation {
                Text("→ \(translation)")
                    .font(horizontalSizeClass == .compact ? .footnote : .body)
                    .foregroundStyle(.orange)
                    .textSelection(.enabled)
            }
        }
        .padding(.vertical, 2)
    }
}

// MARK: - Audio Level Waveform

private struct AudioLevelView: View {
    let waveform: [Float]
    let rms: Float
    let gain: Float

    var body: some View {
        GeometryReader { geo in
            let barWidth = max(2, geo.size.width * 0.008)
            let barSpacing = max(1, barWidth * 0.5)
            let maxBarHeight = geo.size.height * 0.85
            let metricsWidth = min(100, geo.size.width * 0.25)

            HStack(spacing: 0) {
                // Waveform bars
                HStack(alignment: .center, spacing: barSpacing) {
                    ForEach(Array(waveform.enumerated()), id: \.offset) { _, level in
                        let normalized = min(1.0, level * 10)
                        RoundedRectangle(cornerRadius: barWidth * 0.3)
                            .fill(barColor(for: level))
                            .frame(
                                width: barWidth,
                                height: max(2, CGFloat(normalized) * maxBarHeight)
                            )
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .animation(.linear(duration: 0.05), value: waveform)

                // RMS + Gain labels
                VStack(alignment: .trailing, spacing: 2) {
                    Text("RMS \(String(format: "%.4f", rms))")
                        .font(.system(.caption2, design: .monospaced))
                        .foregroundStyle(.secondary)
                    Text("Gain \(String(format: "%.1fx", gain))")
                        .font(.system(.caption2, design: .monospaced))
                        .foregroundStyle(gain > 1.0 ? .orange : .secondary)
                }
                .frame(width: metricsWidth, alignment: .trailing)
            }
        }
    }

    private func barColor(for level: Float) -> Color {
        if level < 0.005 { return .gray.opacity(0.3) }
        if level < 0.02 { return .yellow }
        if level < 0.1 { return .green }
        return .red
    }
}

#Preview {
    ContentView()
}
