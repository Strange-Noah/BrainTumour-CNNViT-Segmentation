import unittest

import nibabel as nib
import numpy as np

from .baseline import instances, propagate, sequences


class BaselineTest(unittest.TestCase):
    def test_label_three_and_diagonal_contact(self):
        raw = np.zeros((5, 5, 5), dtype=np.uint8)
        raw[0, 0, 0] = raw[1, 1, 1] = raw[4, 4, 4] = 3
        raw[2, 2, 2] = 2
        labels = instances(raw)
        np.testing.assert_array_equal(labels > 0, raw == 3)
        self.assertEqual(labels[0, 0, 0], labels[1, 1, 1])
        self.assertEqual(labels.max(), 2)

    def test_affine_translation(self):
        a = np.zeros((5, 5, 5), dtype=np.int32)
        a[2, 2, 2] = 2
        target = np.eye(4)
        target[0, 3] = 1
        ref = nib.Nifti1Image(a * 0, target)
        moved = propagate(a, np.eye(4), ref)
        self.assertEqual(np.count_nonzero(moved), 1)
        self.assertEqual(moved[1, 2, 2], 2)

    def test_sequence_is_ordered_and_duplicate_is_rejected(self):
        rows = [dict(patient_id="x", exam_id=str(i), visit_order=i, relative_days=None)
                for i in (2, 1)]
        self.assertEqual([r["visit_order"] for r in sequences(rows)["x"]], [1, 2])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            sequences(rows + rows[:1])
