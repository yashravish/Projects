vdiagram
========

Vector DB to Mermaid diagram CLI (offline-first). Generate sequence/class/flow diagrams from local code/docs.

Quickstart
----------

```bash
npm install
npx tsx src/cli.ts init
npx tsx src/cli.ts models pull --allow-network
npm run build
node dist/cli.js --help
```

Notes
-----
- Offline-first: no network unless `--allow-network` is provided.
- Windows-friendly; stored paths use POSIX normalization.
- Rendering (PNG/SVG) is optional; Mermaid text export always works.

Commands (MVP)
--------------
- init: setup config and directories
- models pull: cache embedding model locally (optional, network one-time)
- ingest: discover -> chunk -> embed -> index (coming)
- diagram sequence|class|flow: generate Mermaid (coming)
- show | export | stats | reindex | purge (coming)

Requirements
------------
- Node.js >= 18

Cross-platform notes
--------------------
- Windows PowerShell: quote globs as "**/*.ts". Paths are normalized to POSIX in storage.
- macOS/Linux: standard shells work. No symlink traversal outside the project root.

Offline model cache
-------------------
- Default model: sentence-transformers/all-MiniLM-L6-v2 cached under ~/.vdiagram/models.
- Download explicitly once: `vdiagram models pull --allow-network`. Otherwise CLI remains offline.

Examples
--------
- See `examples/minirepo`. Try ingesting and generating sequence/class/flow diagrams there.

Packaging & publish
-------------------
- Build: `npm run build`. Pack: `npm pack`. Global install (local tarball): `npm i -g ./vdiagram-<version>.tgz`.
- Publish: update version in package.json, then `npm publish`.
- Single-binary (optional): prefer JSON store (no native deps). Tools like `pkg`/`nexe` can bundle ESM, but verify dynamic imports.

Performance tuning
------------------
- Ingest flags: `--target-chunk-size 900 --overlap 140 --batch-size 128 --concurrency 2-4`.
- Skip large files: `--max-file-size 1048576` (default 1MB).
- Retrieval defaults: k=24, mmrLambda=0.5, minScore=0.2 (configurable in .vdiagram.json).



