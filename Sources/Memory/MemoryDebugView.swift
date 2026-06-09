import Foundation
import SwiftUI

@MainActor
public struct MemoryDebugView: View {
    private let index: MemoryIndex
    private let pageSize: Int

    @State private var records: [MemoryRecord] = []
    @State private var totalCount = 0
    @State private var isLoading = false
    @State private var isLoadingMore = false
    @State private var errorMessage: String?
    @State private var searchText = ""
    @State private var sort = MemoryDebugSort.createdAtDescending
    @State private var selectedKinds = Set(MemoryKind.allCases)
    @State private var selectedStatuses = Set<MemoryStatus>([.active, .resolved, .superseded])
    @State private var pendingArchive: MemoryRecord?
    @State private var showsArchiveConfirmation = false

    public init(index: MemoryIndex, pageSize: Int = 25) {
        self.index = index
        self.pageSize = max(1, pageSize)
    }

    public var body: some View {
        NavigationStack {
            content
                .navigationTitle("Memories")
                .toolbar {
                    toolbarContent
                }
                .searchable(text: $searchText, prompt: "Search memories")
                .refreshable {
                    await load(reset: true)
                }
                .task(id: querySignature) {
                    await reloadAfterDebounce()
                }
                .navigationDestination(for: MemoryRecord.self) { record in
                    MemoryDebugDetailView(record: record) {
                        confirmArchive(record)
                    }
                }
                .confirmationDialog(
                    "Archive Memory?",
                    isPresented: $showsArchiveConfirmation,
                    titleVisibility: .visible
                ) {
                    Button("Archive", role: .destructive) {
                        Task {
                            await archivePendingMemory()
                        }
                    }
                    Button("Cancel", role: .cancel) {
                        pendingArchive = nil
                    }
                } message: {
                    Text("Archived memories are hidden from the default debug list.")
                }
        }
    }

    @ViewBuilder
    private var content: some View {
        if isLoading && records.isEmpty {
            ProgressView()
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else if records.isEmpty {
            emptyState
        } else {
            memoryList
        }
    }

    private var emptyState: some View {
        VStack(spacing: 16) {
            if let errorMessage {
                ContentUnavailableView(
                    "Unable to Load Memories",
                    systemImage: "exclamationmark.triangle",
                    description: Text(errorMessage)
                )
                Button {
                    Task {
                        await load(reset: true)
                    }
                } label: {
                    Label("Retry", systemImage: "arrow.clockwise")
                }
                .buttonStyle(.borderedProminent)
            } else {
                ContentUnavailableView(
                    searchText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "No Memories" : "No Results",
                    systemImage: "tray",
                    description: Text("No stored memories match the current query.")
                )
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var memoryList: some View {
        List {
            Section {
                ForEach(records, id: \.id) { record in
                    NavigationLink(value: record) {
                        MemoryDebugRow(record: record)
                    }
                    .contextMenu {
                        Button(role: .destructive) {
                            confirmArchive(record)
                        } label: {
                            Label("Archive", systemImage: "archivebox")
                        }
                    }
                    .swipeActions(edge: .trailing, allowsFullSwipe: false) {
                        Button(role: .destructive) {
                            confirmArchive(record)
                        } label: {
                            Label("Archive", systemImage: "archivebox")
                        }
                    }
                }
            } header: {
                Text(totalSummary)
            }

            if records.count < totalCount {
                Section {
                    Button {
                        Task {
                            await load(reset: false)
                        }
                    } label: {
                        HStack {
                            Spacer()
                            if isLoadingMore {
                                ProgressView()
                            } else {
                                Label("Load More", systemImage: "chevron.down")
                            }
                            Spacer()
                        }
                    }
                    .disabled(isLoadingMore)
                }
            }
        }
        .overlay(alignment: .bottom) {
            if let errorMessage, !records.isEmpty {
                Text(errorMessage)
                    .font(.footnote)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(.regularMaterial)
                    .clipShape(Capsule())
                    .padding()
            }
        }
    }

    @ToolbarContentBuilder
    private var toolbarContent: some ToolbarContent {
        ToolbarItemGroup(placement: .primaryAction) {
            Menu {
                Picker("Sort", selection: $sort) {
                    ForEach(MemoryDebugSort.allCases, id: \.self) { value in
                        Text(value.displayTitle).tag(value)
                    }
                }
            } label: {
                Label("Sort", systemImage: "arrow.up.arrow.down")
            }

            Menu {
                Section("Kinds") {
                    ForEach(MemoryKind.allCases, id: \.self) { kind in
                        Toggle(kind.displayTitle, isOn: kindBinding(kind))
                    }
                }

                Section("Statuses") {
                    ForEach(MemoryStatus.allCases, id: \.self) { status in
                        Toggle(status.displayTitle, isOn: statusBinding(status))
                    }
                }
            } label: {
                Label("Filters", systemImage: "line.3.horizontal.decrease.circle")
            }

            Button {
                Task {
                    await load(reset: true)
                }
            } label: {
                Label("Refresh", systemImage: "arrow.clockwise")
            }
            .disabled(isLoading)
        }
    }

    private var querySignature: QuerySignature {
        QuerySignature(
            searchText: searchText,
            sort: sort,
            kinds: selectedKinds,
            statuses: selectedStatuses
        )
    }

    private var totalSummary: String {
        "\(records.count) of \(totalCount)"
    }

    private var activeKindFilter: Set<MemoryKind>? {
        selectedKinds.count == MemoryKind.allCases.count ? nil : selectedKinds
    }

    private var activeStatusFilter: Set<MemoryStatus>? {
        selectedStatuses.count == MemoryStatus.allCases.count ? nil : selectedStatuses
    }

    private func reloadAfterDebounce() async {
        if !searchText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            try? await Task.sleep(nanoseconds: 250_000_000)
        }
        guard !Task.isCancelled else { return }
        await load(reset: true)
    }

    private func load(reset: Bool) async {
        if reset {
            isLoading = true
        } else {
            guard records.count < totalCount, !isLoadingMore else { return }
            isLoadingMore = true
        }

        defer {
            isLoading = false
            isLoadingMore = false
        }

        let offset = reset ? 0 : records.count
        do {
            let page = try await index.debugMemories(
                MemoryDebugQuery(
                    searchText: searchText,
                    limit: pageSize,
                    offset: offset,
                    sort: sort,
                    kinds: activeKindFilter,
                    statuses: activeStatusFilter
                )
            )
            errorMessage = nil
            totalCount = page.totalCount
            if reset {
                records = page.records
            } else {
                records.append(contentsOf: page.records)
            }
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func kindBinding(_ kind: MemoryKind) -> Binding<Bool> {
        Binding {
            selectedKinds.contains(kind)
        } set: { isSelected in
            if isSelected {
                selectedKinds.insert(kind)
            } else {
                selectedKinds.remove(kind)
            }
        }
    }

    private func statusBinding(_ status: MemoryStatus) -> Binding<Bool> {
        Binding {
            selectedStatuses.contains(status)
        } set: { isSelected in
            if isSelected {
                selectedStatuses.insert(status)
            } else {
                selectedStatuses.remove(status)
            }
        }
    }

    private func confirmArchive(_ record: MemoryRecord) {
        pendingArchive = record
        showsArchiveConfirmation = true
    }

    private func archivePendingMemory() async {
        guard let record = pendingArchive else { return }
        pendingArchive = nil
        do {
            _ = try await index.setMemoryStatus(id: record.id, status: .archived)
            await load(reset: true)
        } catch {
            errorMessage = error.localizedDescription
        }
    }
}

private struct QuerySignature: Hashable {
    var searchText: String
    var sort: MemoryDebugSort
    var kinds: Set<MemoryKind>
    var statuses: Set<MemoryStatus>
}

private struct MemoryDebugRow: View {
    let record: MemoryRecord

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Text(record.title ?? record.kind.displayTitle)
                    .font(.headline)
                    .lineLimit(1)
                Spacer(minLength: 8)
                StatusBadge(status: record.status)
            }

            Text(record.text)
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .lineLimit(3)

            HStack(spacing: 12) {
                Label(record.kind.displayTitle, systemImage: "tag")
                Label(record.createdAt.debugFormatted, systemImage: "calendar")
                Label("Importance \(record.importance.debugScoreFormatted)", systemImage: "star")
            }
            .font(.caption)
            .foregroundStyle(.secondary)

            if !record.tags.isEmpty {
                Text(record.tags.prefix(3).map(\.name).joined(separator: ", "))
                    .font(.caption)
                    .foregroundStyle(.tertiary)
                    .lineLimit(1)
            }
        }
        .padding(.vertical, 4)
    }
}

private struct MemoryDebugDetailView: View {
    let record: MemoryRecord
    let archive: () -> Void

    var body: some View {
        Form {
            Section("Memory") {
                Text(record.text)
                    .textSelection(.enabled)
                LabeledContent("Kind", value: record.kind.displayTitle)
                LabeledContent("Status", value: record.status.displayTitle)
                LabeledContent("Importance", value: record.importance.debugScoreFormatted)
                LabeledContent("Confidence", value: record.confidence?.debugScoreFormatted ?? "None")
            }

            Section("Identity") {
                CopyableLabeledContent("ID", value: record.id)
                if let canonicalKey = record.canonicalKey {
                    CopyableLabeledContent("Canonical Key", value: canonicalKey)
                }
                CopyableLabeledContent("Document", value: record.documentPath)
                if !record.source.isEmpty {
                    CopyableLabeledContent("Source", value: record.source)
                }
            }

            Section("Dates") {
                LabeledContent("Created", value: record.createdAt.debugFormatted)
                if let eventAt = record.eventAt {
                    LabeledContent("Event", value: eventAt.debugFormatted)
                }
                LabeledContent("Modified", value: record.modifiedAt.debugFormatted)
                if let lastAccessedAt = record.lastAccessedAt {
                    LabeledContent("Last Accessed", value: lastAccessedAt.debugFormatted)
                }
                LabeledContent("Access Count", value: "\(record.accessCount)")
            }

            Section("Tags") {
                if record.tags.isEmpty && record.facetTags.isEmpty && record.topics.isEmpty {
                    Text("None")
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(record.tags, id: \.self) { tag in
                        LabeledContent(tag.name, value: tag.confidence.debugScoreFormatted)
                    }
                    ForEach(record.facetTags.sorted(by: { $0.rawValue < $1.rawValue }), id: \.self) { facet in
                        LabeledContent("Facet", value: facet.displayTitle)
                    }
                    ForEach(record.topics, id: \.self) { topic in
                        LabeledContent("Topic", value: topic)
                    }
                }
            }

            Section("Entities") {
                if record.entities.isEmpty {
                    Text("None")
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(record.entities, id: \.self) { entity in
                        VStack(alignment: .leading, spacing: 4) {
                            Text(entity.label.displayTitle)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            Text(entity.value)
                                .textSelection(.enabled)
                            if entity.normalizedValue != entity.value {
                                Text(entity.normalizedValue)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                    .textSelection(.enabled)
                            }
                        }
                    }
                }
            }

            Section("Metadata") {
                if record.metadata.isEmpty {
                    Text("None")
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(record.metadata.keys.sorted(), id: \.self) { key in
                        CopyableLabeledContent(key, value: record.metadata[key] ?? "")
                    }
                }
            }

            if let score = record.score {
                Section("Score") {
                    LabeledContent("Semantic", value: score.semantic.debugScoreFormatted)
                    LabeledContent("Lexical", value: score.lexical.debugScoreFormatted)
                    LabeledContent("Recency", value: score.recency.debugScoreFormatted)
                    LabeledContent("Tag", value: score.tag.debugScoreFormatted)
                    LabeledContent("Schema", value: score.schema.debugScoreFormatted)
                    LabeledContent("Temporal", value: score.temporal.debugScoreFormatted)
                    LabeledContent("Status", value: score.status.debugScoreFormatted)
                    LabeledContent("Fused", value: score.fused.debugScoreFormatted)
                    LabeledContent("Rerank", value: score.rerank.debugScoreFormatted)
                    LabeledContent("Blended", value: score.blended.debugScoreFormatted)
                }
            }
        }
        .navigationTitle(record.title ?? record.kind.displayTitle)
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button(role: .destructive, action: archive) {
                    Label("Archive", systemImage: "archivebox")
                }
            }
        }
    }
}

private struct CopyableLabeledContent: View {
    let label: String
    let value: String

    init(_ label: String, value: String) {
        self.label = label
        self.value = value
    }

    var body: some View {
        LabeledContent {
            Text(value)
                .textSelection(.enabled)
        } label: {
            Text(label)
        }
    }
}

private struct StatusBadge: View {
    let status: MemoryStatus

    var body: some View {
        Text(status.displayTitle)
            .font(.caption2)
            .fontWeight(.semibold)
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(status.tint.opacity(0.16), in: Capsule())
            .foregroundStyle(status.tint)
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

    var tint: Color {
        switch self {
        case .active:
            return .green
        case .superseded:
            return .orange
        case .resolved:
            return .blue
        case .archived:
            return .gray
        }
    }
}

private extension FacetTag {
    var displayTitle: String {
        rawValue.replacingOccurrences(of: "_", with: " ").capitalized
    }
}

private extension EntityLabel {
    var displayTitle: String {
        rawValue.replacingOccurrences(of: "_", with: " ").capitalized
    }
}

private extension MemoryDebugSort {
    var displayTitle: String {
        switch self {
        case .createdAtDescending:
            return "Created"
        case .updatedAtDescending:
            return "Updated"
        case .importanceDescending:
            return "Importance"
        case .mostAccessed:
            return "Most Accessed"
        }
    }
}

private extension Date {
    var debugFormatted: String {
        formatted(date: .abbreviated, time: .shortened)
    }
}

private extension Double {
    var debugScoreFormatted: String {
        formatted(.number.precision(.fractionLength(2)))
    }
}
