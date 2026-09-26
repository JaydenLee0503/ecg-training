"""Narrow, hash-checked compatibility for the three pre-rename manifests.

The manifests, scientific protocols and result files are never rewritten.
Both their original source snapshots and the relocated implementations must
match the reviewed migration record. Unknown manifests get no exemption.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MIGRATION = Path(__file__).resolve().parent / 'provenance/descriptive_rename_v1.json'


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_known_migration(manifest, manifest_sha):
    migration = json.loads(MIGRATION.read_text())
    entry = migration['manifests'].get(manifest_sha)
    if entry is None:
        return False
    if manifest['code_sha256'] != entry['original_code_sha256']:
        raise ValueError('Changed original code manifest')
    for name, expected in migration['support_sha256'].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError(f'Changed migration support: {name}')
    for name, expected in entry['original_code_sha256'].items():
        moved = entry['files'][name]
        if (file_hash(ROOT / moved['snapshot']) != expected
                or file_hash(ROOT / moved['current']) != moved['current_sha256']):
            raise ValueError(f'Changed renamed implementation or source snapshot: {name}')
    return True
