import CryptoKit
import Foundation
import AgentMemory
import SwiftUI

@main
struct DebugViewExampleApp: App {
    var body: some Scene {
        WindowGroup {
            DebugViewExampleRoot()
        }
    }
}

private struct DebugViewExampleRoot: View {
    @State private var index: MemoryIndex?
    @State private var errorMessage: String?
    @State private var debugViewID = UUID()

    var body: some View {
        Group {
            if let index {
                TabView {
                    AddMemoryView(index: index, debugViewID: $debugViewID)
                        .tabItem {
                            Label("Add", systemImage: "plus.circle")
                        }

                    MemoryDebugView(index: index)
                        .id(debugViewID)
                        .tabItem {
                            Label("Debug", systemImage: "list.bullet.rectangle")
                        }
                }
            } else if let errorMessage {
                ContentUnavailableView(
                    "Unable to Start",
                    systemImage: "exclamationmark.triangle",
                    description: Text(errorMessage)
                )
            } else {
                ProgressView("Preparing Memory")
            }
        }
        .task {
            await prepareIndexIfNeeded()
        }
    }

    private func prepareIndexIfNeeded() async {
        guard index == nil else { return }

        do {
            let index = try MemoryIndex(
                configuration: MemoryConfiguration(
                    databaseURL: try databaseURL(),
                    embeddingProvider: DemoEmbeddingProvider()
                )
            )
            try await seedIfNeeded(index)
            self.index = index
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func databaseURL() throws -> URL {
        let supportURL = FileManager.default.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        )[0]
            .appendingPathComponent("MemoryDebugViewExample", isDirectory: true)

        try FileManager.default.createDirectory(
            at: supportURL,
            withIntermediateDirectories: true
        )

        return supportURL.appendingPathComponent("memory.sqlite")
    }

    private func seedIfNeeded(_ index: MemoryIndex) async throws {
        let existing = try await index.debugMemories(
            MemoryDebugQuery(limit: 1, statuses: nil)
        )
        guard existing.totalCount == 0 else { return }

        _ = try await index.save(
            text: "The debug view example uses an in-app form to save memories.",
            kind: .fact,
            importance: 0.7,
            source: "debug_view_example",
            tags: ["example", "swiftui"],
            topics: ["debug view"],
            metadata: ["seed": "true"]
        )

        _ = try await index.save(
            text: "Prefer keeping debug tools read-mostly, with archive as the destructive action.",
            kind: .profile,
            importance: 0.8,
            source: "debug_view_example",
            tags: ["debug", "product"],
            facetTags: [.preference],
            metadata: ["seed": "true"]
        )
    }
}

private struct AddMemoryView: View {
    let index: MemoryIndex
    @Binding var debugViewID: UUID

    @State private var text = "Remember that the iOS debug view can inspect AgentMemory records."
    @State private var kind = MemoryKind.fact
    @State private var status = MemoryStatus.active
    @State private var importance = 0.6
    @State private var tagsText = "example, debug"
    @State private var isSaving = false
    @State private var saveMessage: String?

    var body: some View {
        NavigationStack {
            Form {
                Section("Memory") {
                    TextEditor(text: $text)
                        .frame(minHeight: 120)
                    Picker("Kind", selection: $kind) {
                        ForEach(MemoryKind.allCases, id: \.self) { kind in
                            Text(kind.displayTitle).tag(kind)
                        }
                    }
                    Picker("Status", selection: $status) {
                        ForEach(MemoryStatus.allCases, id: \.self) { status in
                            Text(status.displayTitle).tag(status)
                        }
                    }
                    VStack(alignment: .leading) {
                        Text("Importance \(importance.formatted(.number.precision(.fractionLength(2))))")
                        Slider(value: $importance, in: 0...1)
                    }
                    TextField("Tags", text: $tagsText)
                        .textInputAutocapitalization(.never)
                }

                if let saveMessage {
                    Section {
                        Text(saveMessage)
                            .foregroundStyle(.secondary)
                    }
                }

                Section {
                    Button {
                        Task {
                            await saveMemory()
                        }
                    } label: {
                        if isSaving {
                            ProgressView()
                        } else {
                            Label("Save Memory", systemImage: "tray.and.arrow.down")
                        }
                    }
                    .disabled(isSaving || text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }
            }
            .navigationTitle("Add Memory")
        }
    }

    private func saveMemory() async {
        isSaving = true
        defer { isSaving = false }

        do {
            let record = try await index.save(
                text: text.trimmingCharacters(in: .whitespacesAndNewlines),
                kind: kind,
                status: status,
                importance: importance,
                source: "debug_view_example",
                tags: parsedTags,
                metadata: ["created_by": "DebugViewApp"]
            )
            saveMessage = "Saved \(record.kind.displayTitle) memory."
            debugViewID = UUID()
            text = ""
        } catch {
            saveMessage = error.localizedDescription
        }
    }

    private var parsedTags: [String] {
        tagsText
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() }
            .filter { !$0.isEmpty }
    }
}

private actor DemoEmbeddingProvider: EmbeddingProvider {
    let identifier = "debug-view-demo-embedding"
    private let tokenizer = DefaultTokenizer()
    private let dimension = 64

    func embed(texts: [String]) async throws -> [[Float]] {
        texts.map(embedding(for:))
    }

    private func embedding(for text: String) -> [Float] {
        var vector = Array(repeating: Float.zero, count: dimension)

        for token in tokenizer.tokenize(text) {
            let digest = SHA256.hash(data: Data(token.utf8))
            let index = digest.withUnsafeBytes { rawBuffer in
                Int(rawBuffer[0]) % dimension
            }
            vector[index] += 1
        }

        return vector
    }
}

private extension MemoryKind {
    var displayTitle: String {
        rawValue.replacingOccurrences(of: "_", with: " ").capitalized
    }
}

private extension MemoryStatus {
    var displayTitle: String {
        rawValue.replacingOccurrences(of: "_", with: " ").capitalized
    }
}
