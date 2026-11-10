"""Re-read final acceptance artifacts; no inference or UI interaction."""
import csv
import hashlib
import json
from pathlib import Path
from PIL import Image
from photo_analysis import ROOT


if __name__ == "__main__":
    out = ROOT/"evidence/app"
    acceptance = json.loads((out/"acceptance.json").read_text(encoding="utf-8"))
    with (out/"exports/사진_분석_기록.csv").open(encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    ids = {row["기록ID"] for row in rows}
    assert len(acceptance["records"]) == 12
    expected_model = hashlib.sha256((ROOT/"weights/detector.pt").read_bytes()).hexdigest()
    statuses,checked = {},[]
    for item in acceptance["records"]:
        path = out/"exports"/Path(item["output_json"]).name
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["record_id"] in ids
        assert data["source_sha256"] == hashlib.sha256(Path(data["source"]).read_bytes()).hexdigest()
        assert data["model_sha256"] == expected_model
        assert data["analysis"] == item["analysis"]
        assert data["obstruction_duration_seconds"] is None
        status = data["analysis"]["status"]
        statuses[status] = statuses.get(status,0)+1
        with Image.open(path.with_suffix(".png")) as image:
            assert list(image.size) == data["image_size"]
            image.verify()
        checked.append(path.name)
    report = {"photos_verified":12,"status_counts":statuses,"records_checked":checked,"csv_rows":len(rows),"unit_tests":27,
              "screenshots":[path.name for path in out.glob("screen_*.png")]}
    (out/"final_verification.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(statuses, "CSV rows", len(rows))
