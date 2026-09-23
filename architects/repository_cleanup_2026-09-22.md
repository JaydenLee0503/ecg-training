# Repository cleanup — 2026-09-22

The user requested archiving unused scratch work and removing useless files.
The cleanup moved **two files** and removed **34 generated files** with a combined
logical size of **621,947 bytes** (about 607 KiB). The archive move preserves data;
it does not free disk space.

## Archived without changing content

| Original location | Archive location |
|---|---|
| `scratch/wst_dev.py` | `legacy/scratch/wst_dev.py` |
| `scratch/windows.npz` | `legacy/scratch/windows.npz` |

The empty root `scratch/` directory was removed. The interactive script and its
window cache are retained as historical material. The source script and archive
README are eligible for version control; the binary cache stays local and ignored.
The original interactive paths and magics inside the script were preserved, not
converted into a current runnable pipeline.

## Deleted

- **33 Python bytecode files** in the package, scripts, and tests `__pycache__/`
  directories. Python regenerates these from source when needed.
- **`ecgvmd_bundle.zip`**, a generated archive with 11 members, eight of which
  differed from current source. Its generator `make_colab_bundle.py` remains.
  Run `python make_colab_bundle.py` in the project environment when a fresh Colab
  bundle is needed.

The three now-empty bytecode cache directories were also removed. The ignore rules
cover Python/tool caches and distinguish root scratch work from archived source.

## Retained deliberately

- `ECGData.mat`, all feature archives, and all saved experiment outputs/checkpoints.
- `cache/physionet_sources/`: source snippets and metadata supporting the verified
  patient mapping. This directory is audit evidence despite its cache name.
- The active notebooks, package, scripts, tests, and VS Code settings.
- The referenced research PDF and the original notebooks and distinct `.bak`
  snapshots in `legacy/`.
- Frozen experiment protocols, code snapshots, reports, result tables, and figures.

No active code referenced the retired scratch files. Older scripts and results
remain useful for the experiment audit trail and were not classified as disposable
merely because they are old.

## Verification and record

- Checked planned paths and hashes before changes; no source/data file was a
  deletion candidate.
- Checked the saved **930-file VQC improvement inventory** before and after cleanup.
- Compared a SHA-256 digest covering the other preserved repository files before
  and after the operations, excluding Git internals, the planned documentation
  edits, moved files, and deleted generated files.
- Checked that both archived files retain their original hashes, all 34 deletion
  targets are absent, and the old scratch/cache directories are gone.
- Active code was unchanged. Verification focused on file integrity, archive paths,
  documentation links, and ignore behavior; no models were retrained.

The [machine-readable manifest](repository_cleanup_2026-09-22.json) records every
move/deletion with its original size, SHA-256 hash, and deletion reason, plus
documentation changes and integrity results. The shared
[session handoff](SESSION_HANDOFF.md), [archive index](../legacy/README.md), README,
and experiment log were updated.
