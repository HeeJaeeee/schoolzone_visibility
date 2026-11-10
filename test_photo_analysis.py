import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image

from photo_analysis import ROOT, Detector, assess, canonical_detection, export_result, validate_model_names


def vehicle(box):
    return {"kind": "vehicle", "confidence": .9, "box": box}


class AnalysisTests(unittest.TestCase):
    def test_inside_vehicle_is_candidate(self):
        self.assertEqual(assess([vehicle([.2,.2,.4,.4])], [.1,.1,.5,.5])["candidate_count"], 1)

    def test_outside_and_person_are_not_candidates(self):
        result = assess([vehicle([.7,.7,.9,.9]), {"kind":"person", "box":[.1,.1,.5,.5]}], [.1,.1,.5,.5])
        self.assertEqual(result["status"], "no_candidate")

    def test_small_car_fully_inside_large_zone_counts(self):
        # The old rule (overlap >= 10% of zone area) missed this case.
        self.assertEqual(assess([vehicle([.4,.4,.45,.45])], [0,0,1,1])["candidate_count"], 1)

    def test_missing_region_is_unknown(self):
        self.assertEqual(assess([], None)["status"], "region_required")

    def test_empty_detection(self):
        self.assertEqual(assess([], [0,0,1,1])["status"], "no_candidate")

    def test_bad_region_rejected(self):
        for region in ([.4,.1,.3,.2], [0,0,0,1], [-.1,0,1,1], [0,0,float('nan'),1]):
            with self.assertRaises(ValueError):
                assess([], region)

    def test_coco_mapping_excludes_bicycle_and_motorbike(self):
        self.assertIsNone(canonical_detection(1,.8,[1,2,3,4],10,10))
        self.assertIsNone(canonical_detection(3,.8,[1,2,3,4],10,10))
        self.assertEqual(canonical_detection(7,.8,[1,2,3,4],10,10)["kind"], "vehicle")
        self.assertEqual(canonical_detection(0,.8,[1,2,3,4],10,10)["box"], [.1,.2,.3,.4])

    def test_export_real_source_and_json_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "한글 사진.png"
            Image.new("RGB", (100,100), "white").save(source)
            paths = export_result(source, [.1,.1,.5,.5], [vehicle([.2,.2,.4,.4])], {"camera":"검수"}, Path(directory)/"출력")
            data = json.loads(paths[1].read_text(encoding="utf-8"))
            self.assertEqual(data["analysis"]["candidate_count"], 1)
            self.assertEqual(data["image_size"], [100,100])
            self.assertEqual(len(data["source_sha256"]), 64)
            self.assertTrue(paths[0].exists())
            self.assertTrue(paths[2].exists())

    def test_export_rejects_changed_source_before_writing(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/"photo.png"
            Image.new("RGB",(100,100)).save(source)
            output = Path(folder)/"outputs"
            with self.assertRaises(ValueError):
                export_result(source,[0,0,1,1],[],{},output,expected_source_sha="0"*64)
            self.assertFalse(output.exists())

    def test_failed_journal_write_does_not_leave_partial_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/"photo.png"
            Image.new("RGB",(100,100)).save(source)
            output = Path(folder)/"outputs"
            output.mkdir()
            (output/"사진_분석_기록.csv").mkdir()
            with self.assertRaises(OSError):
                export_result(source,[0,0,1,1],[],{},output)
            self.assertEqual(list(output.glob("*.png")),[])
            self.assertEqual(list(output.glob("*.json")),[])

    def test_runtime_uses_declared_class_map(self):
        result = canonical_detection(1,.8,[1,2,3,4],10,10,mapping={"0":0,"1":1})
        self.assertEqual(result["kind"],"vehicle")
        self.assertIsNone(canonical_detection(7,.8,[1,2,3,4],10,10,mapping={"0":0,"1":1}))

    def test_wrong_class_names_fail_clearly(self):
        choice = {"model_class_to_canonical":{"0":0,"2":1},"expected_model_names":{"0":"person","2":"vehicle"}}
        with self.assertRaises(ValueError):
            validate_model_names(choice,{0:"person",2:"bicycle"})
        validate_model_names(choice,{0:"person",2:"vehicle"})

    def test_invalid_model_is_not_cached_for_retry(self):
        choice = json.loads((ROOT/"model_choice.json").read_text(encoding="utf-8"))
        choice["expected_model_names"] = {"2":"incorrect class"}
        detector = Detector(choice)
        for _ in range(2):
            with self.assertRaises(ValueError):
                detector.predict("unused path")
            self.assertIsNone(detector.model)

    def test_export_uses_actual_inference_model_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"source.png"
            Image.new("RGB",(100,100)).save(path)
            choice = {"selected_model":"bound inference model", "weights_sha256":"a"*64,"confidence":.4,"image_size":640}
            outputs = export_result(path,[0,0,1,1],[],{},Path(folder)/"output",model_choice=choice)
            data = json.loads(outputs[1].read_text(encoding="utf-8"))
            self.assertEqual(data["model"],choice["selected_model"])
            self.assertEqual(data["model_sha256"],choice["weights_sha256"])
            self.assertEqual(data["confidence"],.4)
            self.assertEqual(data["image_size_for_inference"],640)


if __name__ == "__main__":
    unittest.main()
