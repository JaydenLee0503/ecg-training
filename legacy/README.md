# legacy

The three original exploratory notebooks, plus the `.bak` copies made before they were
patched. **Nothing in the active pipeline imports from here.**

| file | what it was |
|---|---|
| `new_Arrythmia.ipynb` | the strongest engineering — `Config` dataclass, batched solver, 27-descriptor feature set. Its solver and features are now `ecgvmd/vmd.py` and `ecgvmd/features.py`. |
| `Arrythmia_v2.ipynb` | the strongest science — beat alignment, the alpha question, the chirplet negative result. Its R-peak detector and beat windowing are now `ecgvmd/segment.py`. |
| `Arrythmia_VMD.ipynb` | the weakest, and the only one with genuine correctness bugs (class-sorted sampling, one row per mode). Superseded entirely. |
| `*.ipynb.bak` | pre-patch snapshots, kept so the fixes stay auditable. |

Kept for provenance: every number in the merged pipeline can be traced back to the
notebook it came from. Delete the whole folder if you do not care about that.
