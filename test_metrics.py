import unittest

from metrics import match_counts
from evaluate import source_batches


class MatchingTests(unittest.TestCase):
    def test_evaluation_bounds_4k_image_batches_without_losing_images(self):
        paths = [f"real_image_{i}.jpg" for i in range(9)]
        batches = list(source_batches(paths))
        self.assertLessEqual(max(len(batch) for batch in batches), 4)
        self.assertEqual([p for batch in batches for p in batch], paths)

    # These cases catch inflated detection scores, incorrect class mapping,
    # and missed-object/false-alarm counts when one side is empty.
    def test_exact_box(self):
        self.assertEqual(match_counts([[1, 0.9, 0, 0, 10, 10]], [[1, 0, 0, 10, 10]]), {"tp": 1, "fp": 0, "fn": 0})

    def test_duplicate_detection_counts_once(self):
        self.assertEqual(match_counts([[1, 0.9, 0, 0, 10, 10], [1, 0.8, 0, 0, 10, 10]], [[1, 0, 0, 10, 10]]), {"tp": 1, "fp": 1, "fn": 0})

    def test_wrong_class_does_not_match(self):
        self.assertEqual(match_counts([[0, 0.9, 0, 0, 10, 10]], [[1, 0, 0, 10, 10]]), {"tp": 0, "fp": 1, "fn": 1})

    def test_no_predictions_counts_misses(self):
        self.assertEqual(match_counts([], [[1, 0, 0, 10, 10]]), {"tp": 0, "fp": 0, "fn": 1})

    def test_no_targets_counts_false_alarms(self):
        self.assertEqual(match_counts([[1, 0.9, 0, 0, 10, 10]], []), {"tp": 0, "fp": 1, "fn": 0})

    def test_iou_half_matches_but_lower_overlap_does_not(self):
        target = [[1, 0, 0, 10, 10]]
        self.assertEqual(match_counts([[1, 0.9, 0, 0, 5, 10]], target), {"tp": 1, "fp": 0, "fn": 0})
        self.assertEqual(match_counts([[1, 0.9, 0, 0, 4, 10]], target), {"tp": 0, "fp": 1, "fn": 1})


if __name__ == "__main__":
    unittest.main()
