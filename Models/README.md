# Local model artifacts

Training and evaluation model packages are intentionally not stored in this
repository. Keeping `.mlpackage` directories out of the Git tree allows
SwiftPM and Xcode to resolve AgentMemory without requiring Git LFS.

The package's runtime Core ML model is committed directly under
`Sources/MemoryCoreMLAssets/Resources/` and is available through the
`CoreMLEmbedding` package trait.

To recreate the optional LEAF-IR training artifact used by local evaluation and
autoresearch workflows, run:

```bash
python3 Scripts/convert_leaf_ir_coreml.py
```

That script writes `Models/leaf-ir.mlpackage`. Workflows that require the
embedding baseline at `Models/embedding-v1.mlpackage` can use a local copy of
that generated package. Generated `.mlpackage` directories remain ignored and
must not be committed to this repository.
