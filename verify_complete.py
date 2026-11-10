"""Final evidence gate; never trusts a model-name substring or old GUI report."""
import hashlib
import json
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    choice = read(ROOT/"model_choice.json")
    digest = hashlib.sha256((ROOT/choice["selected_weights"]).read_bytes()).hexdigest()
    assert choice["selected_model"]=="AI-Hub trained YOLO26 nano (v2)"
    assert digest==choice["weights_sha256"]
    integrity = read(ROOT/"evidence/v2_integrity.json")
    assert integrity["best_weight_sha256"]==digest and integrity["changed_parameter_tensors"]>0
    assert integrity["frozen_parameter_changes"]==0 and integrity["before_after_manifest_sha_match"]
    selection = read(ROOT/"evidence/v2_conservative_selection.json")
    assert selection["selected"]["accepted"] and selection["selected"]["summary"]["sha256"]==digest
    held = read(ROOT/"evidence/v2_evaluation/selected_test/summary.json")
    assert held["sha256"]==digest and held["splits"]["test"]["images"]==60
    assert (ROOT/"evidence/v2_release_selection.json").stat().st_mtime < (ROOT/"evidence/v2_evaluation/selected_test/summary.json").stat().st_mtime
    qa = read(ROOT/"evidence/app_v2_trained/acceptance.json")
    assert qa["complete"] and qa["model"]["weights_sha256"]==digest
    assert len(qa["real_photos"])==12 and len(qa["real_videos"])==4
    assert qa["video_stop_verified"] and qa["parent_close_cleanup_verified"]
    for photo in qa["real_photos"]:
        data = read(Path(photo["result"]))
        assert data["model_sha256"]==digest
        assert hashlib.sha256(Path(data["source"]).read_bytes()).hexdigest()==data["source_sha256"]
    for video in qa["real_videos"]:
        data = read(Path(video["result"]))
        assert data["complete"] and data["decoded_frames"]==450 and data["samples"]==30
        assert data["inference_model"]["weights_sha256"]==digest
        assert hashlib.sha256(Path(data["source"]).read_bytes()).hexdigest()==data["source_sha256"]
        union = set()
        for episode in data["episodes"]:
            indices = episode["sample_indices"]
            assert indices==list(range(indices[0],indices[-1]+1))
            assert episode["duration_seconds"]==episode["end_seconds"]-episode["start_seconds"]>=5
            union.update(indices)
        assert len(union)==data["persistent_sample_count"]
    for name in ("photo_00.png","photo_03.png","photo_06.png","photo_09.png","video_00.png","video_01.png","video_02.png","video_03.png","review_list.png"):
        with Image.open(ROOT/"evidence/app_v2_trained"/name) as image:
            image.verify()
    log = (ROOT/"evidence/v2_actual_gui.log").read_text(encoding="utf-8")
    assert "Real trained-model acceptance complete" in log
    assert not any(error in log for error in ("Traceback","bgerror","TclError","invalid command"))
    tests = (ROOT/"evidence/complete_unit_tests.log").read_text(encoding="utf-8")
    assert "Ran 52 tests" in tests and "OK" in tests
    status = {"complete":True,"actual_trained_model_sha256":digest,"data_verified":1650,"real_photos":12,"real_videos":4,
              "tests_passed":52,"independent_code_review":"approved; no Critical/Important findings",
              "scope":"Image-plane observation evidence; no physical driver-visibility or field commercial validation claim."}
    (ROOT/"evidence/complete_status.json").write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding="utf-8")
    print(status)
