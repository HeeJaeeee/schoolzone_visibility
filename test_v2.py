import unittest
from pathlib import Path
from prepare_v2 import split_for_video, label_class


class V2Tests(unittest.TestCase):
    def test_whole_episode_separated(self):
        for view in ("A","B"):
            self.assertEqual(split_for_video(f"2020_11_12_12_30_suddenlyappear_sun_{view}_2.mp4","train1"),"test")

    def test_truck_crossview_is_not_training(self):
        self.assertEqual(split_for_video("2020_11_13_18_30_suddenlyappear_night_A_11.mp4","train4"),"train")
        self.assertEqual(split_for_video("2020_11_13_18_30_suddenlyappear_night_B_11.mp4","train4"),"crossview")

    def test_official_validation_and_background_train(self):
        self.assertEqual(split_for_video("other.mp4","validation"),"development")
        self.assertEqual(split_for_video("other.mp4","train1"),"train")

    def test_only_original_coarse_truth_used(self):
        self.assertEqual(label_class("vehicle"),2)
        self.assertEqual(label_class("person"),0)
        self.assertIsNone(label_class("motorbike"))


if __name__ == "__main__":
    unittest.main()
