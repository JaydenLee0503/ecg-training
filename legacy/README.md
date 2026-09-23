# legacy

The three original exploratory notebooks, their pre-patch `.bak` copies, and retired
scratch work. **Nothing in the active pipeline imports from here.**

| file | what it was |
|---|---|
| `new_Arrythmia.ipynb` | the strongest engineering — `Config` dataclass, batched solver, 27-descriptor feature set. Its solver and features are now `ecgvmd/vmd.py` and `ecgvmd/features.py`. |
| `Arrythmia_v2.ipynb` | the strongest science — beat alignment, the alpha question, the chirplet negative result. Its R-peak detector and beat windowing are now `ecgvmd/segment.py`. |
| `Arrythmia_VMD.ipynb` | the weakest, and the only one with genuine correctness bugs (class-sorted sampling, one row per mode). Superseded entirely. |
| `*.ipynb.bak` | pre-patch snapshots, kept so the fixes stay auditable. |
| [`scratch/`](scratch/README.md) | retired interactive WST exploration and its original local window cache, moved here on 2026-09-22. |

These files preserve the development history; they are not the current evaluation
protocol. The notebook backups differ from their later versions and are retained.
Use the corrected patient-grouped reports in `architects/` for current results.
See the [cleanup record](../architects/repository_cleanup_2026-09-22.md) for the
archive moves, removed generated files, and integrity checks.
