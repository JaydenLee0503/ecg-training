# ACS download verification — 2026-09-23

Paths and commands updated for the ACS workspace reorganization; scientific
results and machine-readable historical records are unchanged. See
[the workspace guide](../README.md).

## Outcome

**The user's two downloads exactly match the official Figshare v1 files. No
replacement download is needed.** Every archive member passed CRC verification.
The source release has WFDB header inconsistencies and several signal-quality
flags, so download integrity passes while readiness for training needs additional
loader/protocol work. No archive was modified or extracted, and no model was trained.

Dataset: *A large-scale 12-lead electrocardiogram dataset for acute coronary
syndrome prediction containing 19,955 ECGs*,
[Figshare v1](https://doi.org/10.6084/m9.figshare.29925314.v1).
The [paper](https://doi.org/10.1038/s41597-026-07278-0) describes the cohort and
the hidden-label test evaluation. This verification does not validate clinical
diagnoses or establish research novelty.

## Download integrity

| Archive | Bytes | Official and computed MD5 | Result |
|---|---:|---|---|
| `experiments/acs/data/CSV.zip` | 296,769 | `1cb46279c0e68e6512bd39214fc56528` | Match |
| `experiments/acs/data/ECG_row_data.zip` | 1,277,577,332 | `acea6ca86a2d0b937ecfe7d1df6d30a2` | Match |

The optional median-waveform archive was not downloaded and is not required to
verify the raw-waveform VMD/WST input. The downloaded CSV archive contains two CSV
files. The waveform archive contains **19,955 `.dat` / `.hea` pairs**. All
**39,912 files** passed ZIP CRC checks; each metadata record references exactly
one available waveform, with no duplicate record references or missing files.
All 19,955 signal files were decoded using the declared 12-channel, multiplexed,
little-endian 16-bit format. No identical raw `.dat` payloads were found.

The source API response, local SHA-256 values, environment, script hash, checks,
and record-specific flags are saved in the linked JSON files below. The final
full verification took **37.497 seconds**, finishing at
**2026-09-23 04:16:45 UTC**. The data archives are ignored by `/data/` in
`.gitignore`; `git ls-files data` returned no tracked files.

## Cohort and supplied split

| Partition | ECG records | Unique patients | Disease labels |
|---|---:|---:|---|
| Training | 17,960 | 17,018 | Provided |
| Test | 1,995 | 1,891 | Withheld |
| Total | 19,955 | 18,909 | — |

**No patient identifier appears in both supplied partitions.** No empty CSV
cells were found. Some patients have multiple ECGs: any further training /
validation split must still group by `Patient_id`. Test demographics and signal
structure were checked only for integrity; no test disease labels were obtained,
inferred, or submitted to an evaluation service.

Training-record labels (before any exclusions):

| Label | Positive | Negative |
|---|---:|---:|
| AMI | 2,679 | 15,281 |
| OMI | 1,151 | 16,809 |
| CTO | 957 | 17,003 |
| NSTEMI | 1,235 | 16,725 |
| STEMI | 1,442 | 16,518 |
| UA | 6,213 | 11,747 |

These are separate binary fields, not mutually exclusive classes. They are
record counts, not counts of independent positive patients. OMI positives are
only **6.41%** of training records; an always-negative classifier obtains
**93.59% training accuracy** without detecting any OMI. A future OMI protocol
should emphasize sensitivity, specificity, balanced accuracy, and precision-recall
performance alongside uncertainty. No target or exclusion policy has been frozen.

## Source-release issues

### Header fields contain signal extrema

Across **all 19,955 records and all 12 channels**, the header field defined by
WFDB as the initial sample instead equals the channel maximum; the field defined
as the checksum instead equals the channel minimum. Every record fails the
standard first-sample and modulo-65,536 checksum checks on at least one channel.
The maximum/minimum correspondence was checked across the entire cohort, not
inferred from a single example.

This follows the published [WFDB header specification](https://wfdb.io/spec/header-files.html)
and [signal encoding specification](https://wfdb.io/spec/signal-files.html).
For example, `00001`, lead I, has initial-value field **892**, while the stored
first sample is **4**; its channel maximum is **892**. The archive hashes prove
that these fields are present in the official release rather than introduced by
the user's download. The audit uses NumPy and an explicit format-16 decoder;
the project's Python environment does not currently have the `wfdb` package.

A future loader should preserve the original archives and explicitly account
for these metadata errors. If corrected WFDB headers are created, place them in
a separate derived directory with a record of the corrections; do not overwrite
or silently repair the source files. The audit did not modify any headers.

### Two records contain fewer samples than declared

All headers declare 12 leads, 500 Hz, and 5,000 samples per lead (10 seconds).
Two training signals instead contain 84,000 bytes, representing 3,500 complete
12-lead frames (7 seconds) under their declared encoding, rather than the expected
120,000 bytes. Their channel extrema still match the corresponding header fields.

| Record | Patient | Stored duration | Partition |
|---|---|---:|---|
| `03228` | `P15456` | 7 s | Training |
| `14262` | `P08823` | 7 s | Training |

Both have supplied AMI=0 and OMI=0 labels. Their source headers have not been
changed. For a fixed 10-second experiment, explicitly exclude these records or
predefine a different duration policy before feature extraction; do not silently
pad them and count them as complete 10-second observations. No exclusion has
actually been applied in this verification.

### Signal-quality flags

- **52 records** (50 train, 2 test) contain at least one lead that is constant
  throughout the stored recording. This does not mean that all leads are flat.
- **3 training records** (`02008`, `03054`, `16558`) each contain one `-32768`
  format-16 missing-value marker. A waveform loader must handle these explicitly
  before VMD/WST feature computation.
- The JSON includes every flagged record, patient, split, and affected constant
  lead. The tests cover exact flatlines and missing markers only; they do not
  establish overall diagnostic signal quality or verify clinical labels.

## Reproduction and execution notes

Run from the repository root using the existing NumPy environment:

```bash
/home/jaydenlee/venvs/test-ecg-training/bin/python -B experiments/acs/scripts/verify_download.py --output experiments/acs/reports/NEW_VERIFICATION.json
```

The script reads the local archives and the saved official source manifest. It
performs no network downloads and writes only its audit JSON. It intentionally
returns **exit code 1** while the known header checks fail: see
`download_integrity: passed` separately from `status: issues_found`. Do not treat
that exit code as a request to re-download byte-identical files.

An initial scan finished but could not save its report because the shell sandbox
rejected the repository write as read-only. It was rerun with approved write
permission. The first saved scan treated the two byte-length discrepancies as
undecodable records; follow-up inspection decoded their complete stored frames
without padding. A final full scan added the cohort-wide extrema check and
confirmed both the source-format findings and all archive checks. The current
JSON records the final code hash and results. No dependencies were installed.

- [Verification results and flagged records](acs_download_verification_2026-09-23.json)
- [Official Figshare API snapshot](acs_download_verification_2026-09-23_source.json)
- [Reproducible verifier](../scripts/verify_download.py)

Next work is a documented ACS loader and patient-separated task protocol that
handles these source issues, followed by matched VMD/WST classical controls and
VQC experiments. The existing ECGData results and frozen protocols remain intact.
