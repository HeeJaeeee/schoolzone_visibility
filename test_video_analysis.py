import tempfile
import json
import threading
import unittest
from unittest.mock import patch
from pathlib import Path
import cv2
import numpy as np
from video_analysis import analyze_video, export_review_table, export_video_report


class StaticDetector:
    choice = {"selected_model":"QA deterministic detector", "weights_sha256":"test"}

    def predict(self,path,snapshot=None):
        return [{"kind":"vehicle","confidence":.9,"box":[.2,.2,.4,.4]}]


class VideoTests(unittest.TestCase):
    def video(self,folder,seconds=7):
        path = Path(folder)/"QA_only.avi"
        writer = cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"MJPG"),10,(100,100))
        self.assertTrue(writer.isOpened())
        for index in range(seconds*10):
            writer.write(np.zeros((100,100,3),dtype=np.uint8))
        writer.release()
        return path

    def test_frame_times_and_alert(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.video(folder)
            frames = []
            result = analyze_video(path,[.1,.1,.5,.5],StaticDetector(),alert_seconds=5,on_frame=lambda image,detections,state:frames.append(state["time"]))
            self.assertEqual(result["samples"],35)  # 10 fps source sampled at 5 fps
            self.assertEqual(frames[:3],[0,.2,.4])
            self.assertEqual(result["alert_count"],1)
            self.assertAlmostEqual(result["alerts"][0]["alert_seconds"],5)
            self.assertEqual(result["decoded_frames"],70)
            self.assertTrue(result["complete"])
            self.assertEqual(len(result["source_sha256"]),64)

    def test_export_writes_snapshot_and_csv(self):
        with tempfile.TemporaryDirectory() as folder:
            report = analyze_video(self.video(folder),[.1,.1,.5,.5],StaticDetector(),alert_seconds=5)
            saved = json.loads(export_video_report(report,{"camera":"QA"},Path(folder)/"out").read_text(encoding="utf-8"))
            self.assertEqual(len(saved["alert_images"]),1)
            self.assertTrue(Path(saved["alert_images"][0]).exists())
            rows = (Path(folder)/"out"/"알림_기록.csv").read_text(encoding="utf-8-sig").splitlines()
            self.assertEqual(len(rows),2)

    def test_cancel_does_not_emit_complete_report(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.video(folder)
            cancel = threading.Event()
            cancel.set()
            with self.assertRaises(InterruptedError):
                analyze_video(path,[.1,.1,.5,.5],StaticDetector(),cancel=cancel)

    def test_cancel_during_final_write_cleans_all_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            source = self.video(folder)
            report = analyze_video(source,[.1,.1,.5,.5],StaticDetector(),alert_seconds=5)
            cancel = threading.Event()
            write = Path.write_text
            def final_write(path,*args,**kwargs):
                result = write(path,*args,**kwargs)
                cancel.set()
                return result
            output = Path(folder)/"output"
            with patch.object(Path,"write_text",final_write):
                with self.assertRaises(InterruptedError):
                    export_video_report(report,{},output,cancel=cancel)
            self.assertEqual(list(output.iterdir()),[])

    def test_bad_input_not_silently_empty(self):
        with self.assertRaises((ValueError,FileNotFoundError)):
            analyze_video(Path("not_a_video.mp4"),[0,0,1,1],StaticDetector())

    def test_review_rejects_mixed_models_and_rules(self):
        base = {"source":"x/a.mp4","source_duration_seconds":30,"alert_count":1,"longest_dwell_seconds":12,"alerts":[{"alert_seconds":10.2}],
                "metadata":{"camera":"A"},"rules":{"alert_seconds":10},"waiting_configured":False,
                "inference_model":{"weights_sha256":"a","confidence":.25,"image_size":1280}}
        with tempfile.TemporaryDirectory() as folder:
            first,second = Path(folder)/"a.json",Path(folder)/"b.json"
            first.write_text(json.dumps(base),encoding="utf-8")
            for altered in ({**base,"rules":{"alert_seconds":5}},{**base,"inference_model":{**base["inference_model"],"weights_sha256":"b"}}):
                second.write_text(json.dumps(altered),encoding="utf-8")
                with self.assertRaises(ValueError):
                    export_review_table([first,second],Path(folder)/"output")
            rows = export_review_table([first,first],Path(folder)/"output")
            self.assertEqual((len(rows),rows[0]["first_alert_seconds"]),(1,10.2))


if __name__ == "__main__":
    unittest.main()
