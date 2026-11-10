import json
import tempfile
import tkinter as tk
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from pathlib import Path
from PIL import Image
from app import PhotoApp
from photo_analysis import DEMO


class GuiTests(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.app = PhotoApp(self.root)
        self.root.update()

    def tearDown(self):
        self.root.destroy()

    def test_demo_has_camera_specific_region(self):
        self.app.open_path(DEMO/self.app.catalog[0]["file"])
        self.assertIsNotNone(self.app.region)
        self.assertIsNone(self.app.detections)

    def test_external_photo_does_not_inherit_demo_region(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"新 사진.png"
            Image.new("RGB",(100,100)).save(path)
            self.app.open_path(path)
            self.assertIsNone(self.app.region)
            self.assertIsNone(self.app.detections)

    def test_region_update_clears_result(self):
        self.app.detections = []
        self.app.apply_region([.1,.1,.4,.4])
        self.assertIsNone(self.app.detections)
        self.assertNotIn("region_note",self.app.metadata)

    def test_display_coordinates_with_letterboxing(self):
        self.app.display_box = (100,200,800,450)
        self.assertEqual(self.app.to_normalized(500,425),(.5,.5))
        self.assertEqual(self.app.to_normalized(0,0),(0,0))

    def test_save_before_analysis_rejected(self):
        with self.assertRaises(ValueError):
            self.app.save_result()

    def test_profile_requires_same_camera_confirmation_and_dimensions(self):
        with tempfile.TemporaryDirectory() as folder:
            profile = Path(folder)/"profile.json"
            profile.write_text(json.dumps({"image_size":list(self.app.image.size), "region":[.1,.1,.4,.4], "camera":"test"}))
            with self.assertRaises(ValueError):
                self.app.load_profile(profile,confirmed=False)
            self.app.load_profile(profile,confirmed=True)
            self.assertEqual(self.app.region,[[.1,.1],[.4,.1],[.4,.4],[.1,.4]])
            profile.write_text(json.dumps({"image_size":[1,1],"region":[0,0,1,1],"camera":"wrong"}))
            with self.assertRaises(ValueError):
                self.app.load_profile(profile,confirmed=True)

    def test_failed_open_preserves_current_state(self):
        old = self.app.path
        with self.assertRaises(FileNotFoundError):
            self.app.open_path(Path("does_not_exist.png"))
        self.assertEqual(self.app.path,old)

    def test_changed_photo_blocks_analysis_and_export(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"変更.png"
            Image.new("RGB",(100,100),"white").save(path)
            self.app.open_path(path)
            self.app.apply_region([.1,.1,.5,.5])
            self.app.detections = []
            Image.new("RGB",(100,100),"black").save(path)
            with self.assertRaises(ValueError):
                self.app.save_result(Path(folder)/"outputs")
            self.app.run_analysis()
            self.assertFalse(self.app.busy)
            self.assertIsNone(self.app.detections)

    def test_deleted_photo_invalidates_stale_result(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"photo.png"
            Image.new("RGB",(100,100)).save(path)
            self.app.open_path(path)
            self.app.apply_region([.1,.1,.5,.5])
            self.app.detections = []
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                self.app.save_result(Path(folder)/"outputs")
            self.assertIsNone(self.app.detections)

    def test_worker_failure_restores_controls_and_blocks_save(self):
        self.app.busy = True
        self.app.messages.put((None,"검수용 추론 오류"))
        self.app.poll()
        self.assertFalse(self.app.busy)
        self.assertIn("분석 실패",self.app.status.get())
        self.assertEqual(str(self.app.save_button["state"]),"disabled")
        self.assertEqual(str(self.app.listbox["state"]),"normal")

    def test_registered_source_video_remains_available_after_region_edit(self):
        self.assertTrue(self.app.registered_video().exists())
        self.app.apply_region([.1,.1,.5,.5])
        self.assertIsNotNone(self.app.registered_video())

    def test_external_photo_has_no_inferred_source_video(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"photo.png"
            Image.new("RGB",(100,100)).save(path)
            self.app.open_path(path)
            self.assertIsNone(self.app.registered_video())

    def test_opened_video_is_monitored_directly(self):
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"cctv.avi"
            writer = cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"MJPG"),5,(64,48))
            writer.write(np.zeros((48,64,3),dtype=np.uint8))
            writer.release()
            self.app.open_path(path)
            self.assertEqual(self.app.image.size,(64,48))
            self.assertEqual(self.app.registered_video(),path.resolve())
            self.assertEqual(str(self.app.video_button["state"]),"disabled")  # no zone yet
            self.app.apply_region([[.1,.1],[.9,.1],[.5,.9]])
            self.assertEqual(str(self.app.video_button["state"]),"normal")

    def test_polygon_drawing_by_clicks(self):
        self.app.display_box = (0,0,100,100)
        self.app.start_edit("entry")
        for x,y in ((10,10),(90,10),(90,90)):
            self.app.click(SimpleNamespace(x=x,y=y))
        self.app.finish_polygon()
        self.assertEqual(self.app.region,[[.1,.1],[.9,.1],[.9,.9]])
        self.assertFalse(self.app.editing)

    def test_main_close_requests_child_cleanup_before_destroy(self):
        close = Mock()
        self.app.video_window = SimpleNamespace(window=self.root,close=close)
        job = self.app.poll_job
        self.app.close()
        close.assert_called_once()
        self.assertTrue(self.app.closing)
        self.assertTrue(self.root.winfo_exists())
        self.assertNotIn(job,self.root.tk.call("after","info"))


if __name__ == "__main__":
    unittest.main()
