"""ACS source integrity, physical decoding, exclusions, and label boundaries."""
import csv
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from experiments.acs.loader import ACSDataset, ACSError, ACSExcluded, LEADS


BASE = ['Patient_id', 'ecg_row_record', 'ecg_med_record', 'gender', 'age', 'Time_Interval']
LABELS = ['AMI', 'OMI', 'CTO', 'PCI', 'NSTEMI', 'STEMI', 'UA', 'LM', 'PLAD',
          'MLAD', 'DLAD', 'DB', 'PLCX', 'MLCX', 'DLCX', 'OM', 'PRCA', 'MRCA',
          'DRCA', 'VF_VT', 'Paced', 'Prior_PCI']


def fixture(root, *, issue=None, standard_header=False, reverse_leads=False):
    """Tiny independent WFDB writer with known signals and repeated train patient."""
    signals = {}
    rows = {'train': [], 'test': []}
    with ZipFile(root / 'ECG_row_data.zip', 'w', ZIP_DEFLATED) as archive:
        for k in range(1, 4):
            record = f'{k:05d}'
            n = 3500 if issue == 'short' and k == 1 else 5000
            data = ((np.arange(n)[:, None] % 61 - 30) * np.arange(1, 13)[None, :] + 10).astype('<i2')
            if issue == 'flat' and k == 1:
                data[:, 6] = 12  # V1, independent of requested lead II.
            if issue == 'missing' and k == 1:
                data[100, 1] = -32768
            names = list(LEADS)
            if reverse_leads:
                data, names = data[:, ::-1].copy(), names[::-1]
            signals[record] = (data.copy(), names)
            initial = data[0] if standard_header else data.max(axis=0)
            checks = ((data.sum(axis=0, dtype=np.int64) + 32768) % 65536 - 32768) if standard_header else data.min(axis=0)
            lines = [f'{record} 12 500 5000']
            for i, name in enumerate(names):
                gain = '2000.0(10)/mV'
                if issue == 'bad_gain' and k == 1 and i == 0:
                    gain = '0.0(10)/mV'
                first = int(initial[i]) + (123 if issue == 'bad_header' and k == 1 else 0)
                lines.append(f'{record}.dat 16 {gain} 16 0 {first} {int(checks[i])} 0 {name}')
            archive.writestr(f'row_data/{record}.hea', '\n'.join(lines) + '\n')
            raw = data.tobytes()
            if issue == 'partial_frame' and k == 1:
                raw = raw[:-1]
            archive.writestr(f'row_data/{record}.dat', raw)
            split = 'test' if k == 3 else 'train'
            patient = 'P00001' if k < 3 or issue == 'patient_leak' else 'P00002'
            row = dict(zip(BASE, [patient, record + '.dat', record + '.med', '1', '55', '60']))
            if split == 'train':
                row.update({key: '0' for key in LABELS})
                row['OMI'] = '1' if k == 1 else '0'
                if issue == 'bad_label' and k == 1:
                    row['OMI'] = ''
                if issue == 'nonbinary_target' and k == 1:
                    row['OMI'] = '2'
                if issue == 'vessel_category':
                    row['PLAD'] = '2'
            elif issue == 'test_labels':
                row['OMI'] = '0'
            rows[split].append(row)
    with ZipFile(root / 'CSV.zip', 'w', ZIP_DEFLATED) as archive:
        for split, partition in rows.items():
            if issue == 'duplicate_record' and split == 'train':
                partition.append(partition[0])
            stream = io.StringIO()
            writer = csv.DictWriter(stream, fieldnames=list(partition[0]))
            writer.writeheader()
            writer.writerows(partition)
            archive.writestr(f'CSV/{split}.csv', stream.getvalue())
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.zip')}
    return hashes, signals


class ACSTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def make(self, **kwargs):
        return fixture(self.root, **kwargs)

    def test_signed_multiplexed_decode_scale_baseline_and_lead_order(self):
        hashes, signals = self.make(reverse_leads=True)
        with ACSDataset(self.root, target='OMI', leads=('V2', 'II'), expected_sha256=hashes) as ds:
            record = ds.load_record('00001')
            raw, names = signals['00001']
            expected = (raw[:, [names.index('V2'), names.index('II')]].astype(float) - 10) / 2000
            np.testing.assert_array_equal(record.signal_mV, expected)
            np.testing.assert_array_equal(record.lead('II'), expected[:, 1])
            self.assertEqual(record.fs, 500)
            self.assertEqual(record.info.label, 1)
            self.assertEqual(record.info.patient_id, ds.info('00002').patient_id)
            self.assertEqual(record.decision.source_header_convention, 'acs_extrema_in_initial_and_checksum_fields')
            self.assertFalse(record.signal_mV.flags.writeable)
            with self.assertRaises(KeyError):
                record.lead('V6')

    def test_standard_header_supported_and_originals_unchanged(self):
        hashes, _ = self.make(standard_header=True)
        with ACSDataset(self.root, expected_sha256=hashes) as ds:
            self.assertEqual(ds.load_record('00001').decision.source_header_convention, 'wfdb_standard')
        self.assertEqual(hashes, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.root.glob('*.zip')})

    def test_test_labels_never_become_negative_training_labels(self):
        hashes, _ = self.make()
        with ACSDataset(self.root, target='OMI', expected_sha256=hashes) as ds:
            self.assertIsNone(ds.load_record('00003').info.label)
            self.assertEqual([r.info.record_id for r in ds.iter_records()], ['00001', '00002'])
            self.assertIsNone(ds.summary()['splits']['test']['eligible_label_counts'])
        with ACSDataset(self.root, expected_sha256=hashes) as ds:
            self.assertIsNone(ds.info('00001').label)

    def test_metadata_rejects_patient_leakage_and_label_errors(self):
        for issue in ('patient_leak', 'bad_label', 'nonbinary_target', 'test_labels', 'duplicate_record'):
            with self.subTest(issue=issue):
                hashes, _ = self.make(issue=issue)
                with self.assertRaises(ACSError):
                    ACSDataset(self.root, target='OMI', expected_sha256=hashes)

    def test_nonbinary_vessel_annotations_do_not_change_diagnosis_labels(self):
        hashes, _ = self.make(issue='vessel_category')
        with ACSDataset(self.root, target='OMI', expected_sha256=hashes) as ds:
            self.assertEqual(ds.load_record('00001').info.label, 1)
            self.assertEqual(ds.load_record('00002').info.label, 0)

    def test_corrupted_archive_fails_identity_before_decoding(self):
        hashes, _ = self.make()
        with (self.root / 'ECG_row_data.zip').open('ab') as stream:
            stream.write(b'unexpected')
        with self.assertRaisesRegex(ACSError, 'SHA-256'):
            ACSDataset(self.root, expected_sha256=hashes)

    def test_short_record_excluded_without_padding_and_logged(self):
        hashes, _ = self.make(issue='short')
        with ACSDataset(self.root, expected_sha256=hashes) as ds:
            record = ds.inspect('00001')
            self.assertIsNone(record.signal_mV)
            self.assertEqual(record.decision.actual_samples, 3500)
            with self.assertRaises(ACSExcluded):
                ds.load_record('00001')
            self.assertEqual([r.info.record_id for r in ds.iter_records()], ['00002'])
            self.assertEqual(ds.summary()['splits']['train']['excluded_records'], 1)
            self.assertEqual(len(ds.decisions), 2)

    def test_missing_marker_not_converted_to_extreme_amplitude(self):
        hashes, _ = self.make(issue='missing')
        with ACSDataset(self.root, leads=('II',), expected_sha256=hashes) as ds:
            with self.assertRaises(ACSExcluded) as raised:
                ds.load_record('00001')
            self.assertIn('missing_requested_sample', raised.exception.decision.exclusion_reasons)
        with ACSDataset(self.root, leads=('I',), expected_sha256=hashes) as ds:
            r = ds.load_record('00001')
            self.assertEqual(r.decision.missing_leads, ('II',))
            self.assertTrue(np.isfinite(r.signal_mV).all())

    def test_flat_lead_exclusion_respects_selected_leads(self):
        hashes, _ = self.make(issue='flat')
        with ACSDataset(self.root, expected_sha256=hashes) as ds:
            with self.assertRaises(ACSExcluded):
                ds.load_record('00001')
        with ACSDataset(self.root, leads=('II',), expected_sha256=hashes) as ds:
            self.assertEqual(ds.load_record('00001').decision.flat_leads, ('V1',))

    def test_unknown_header_corruption_and_partial_frames_raise(self):
        for issue in ('bad_header', 'bad_gain', 'partial_frame'):
            with self.subTest(issue=issue):
                hashes, _ = self.make(issue=issue)
                with ACSDataset(self.root, expected_sha256=hashes) as ds:
                    with self.assertRaises(ACSError):
                        ds.inspect('00001')

    def test_invalid_api_options_and_closed_dataset(self):
        hashes, _ = self.make()
        for leads in ('II', (), ('II', 'II'), ('bad',)):
            with self.assertRaises(ValueError):
                ACSDataset(self.root, leads=leads, expected_sha256=hashes)
        with self.assertRaises(ValueError):
            ACSDataset(self.root, target='unknown', expected_sha256=hashes)
        with ACSDataset(self.root, expected_sha256=hashes) as ds:
            with self.assertRaises(ValueError):
                ds.record_ids('validation')
        with self.assertRaisesRegex(ACSError, 'closed'):
            ds.load_record('00001')


if __name__ == '__main__':
    unittest.main()
