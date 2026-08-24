#!/usr/bin/env python
"""Zip up everything Colab needs, so the notebooks run there unchanged.

    python make_colab_bundle.py

Produces `ecgvmd_bundle.zip`. Upload it when a notebook's bootstrap cell asks, or drop
it in Google Drive at `MyDrive/test-ecg-training/` and the bootstrap will find it.

ECGData.mat is NOT included (70 MB). Upload that separately, or better, put it in Drive
once and mount the drive - re-uploading 70 MB every session gets old fast.
"""
from __future__ import annotations

import os
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "ecgvmd_bundle.zip")

INCLUDE_DIRS = ["ecgvmd", "scripts"]
INCLUDE_FILES = ["run_pipeline.py", "requirements.txt", "README.md"]


def main():
    n = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for d in INCLUDE_DIRS:
            src = os.path.join(ROOT, d)
            if not os.path.isdir(src):
                continue
            for dirpath, dirnames, filenames in os.walk(src):
                dirnames[:] = [x for x in dirnames if x != "__pycache__"]
                for f in filenames:
                    if f.endswith((".pyc", ".npz")):
                        continue
                    full = os.path.join(dirpath, f)
                    z.write(full, os.path.relpath(full, ROOT))
                    n += 1
        for f in INCLUDE_FILES:
            full = os.path.join(ROOT, f)
            if os.path.exists(full):
                z.write(full, f)
                n += 1

    print(f"wrote {OUT}  ({os.path.getsize(OUT) / 1e3:.0f} kB, {n} files)")
    print("\nIn Colab, either:")
    print("  * upload this zip when the notebook's bootstrap cell prompts, or")
    print("  * unzip it to Drive at MyDrive/test-ecg-training/ (with ECGData.mat)")
    print("    and the bootstrap will find it with no upload at all.")


if __name__ == "__main__":
    main()
