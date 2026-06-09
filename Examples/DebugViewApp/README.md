# DebugViewApp

`DebugViewApp` is a minimal iOS SwiftUI example for the `Memory` debug view API.

It has two tabs:

- Add: saves a memory into a local SQLite database using `MemoryIndex.save`.
- Debug: mounts `MemoryDebugView(index:)` so you can search, page, inspect metadata, and archive memories.

Generate/open the project:

```bash
cd Examples/DebugViewApp
xcodegen generate
open DebugViewApp.xcodeproj
```

The example intentionally uses a tiny local embedding provider so it can run without setting up NaturalLanguage or CoreML models.
