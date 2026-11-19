"""Export FP32 and calibrated INT8 LiteRT models from the same checkpoint."""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

import yaml
from ultralytics import YOLO


def export_pair(model_path, calibration, output, size):
    if not any((calibration / "images").glob("*.jpg")):
        raise ValueError(f"No calibration JPEGs under {calibration / 'images'}")
    output.mkdir(parents=True, exist_ok=True)
    artifacts = []
    with tempfile.TemporaryDirectory(dir=output) as work:
        work = Path(work)
        local_model = work / "wearable.pt"
        shutil.copyfile(model_path, local_model)
        config = dict(path=calibration.resolve().as_posix(), train="images", val="images",
                      names=YOLO(local_model).names)
        data = work / "calibration.yaml"
        data.write_text(yaml.safe_dump(config), encoding="utf-8")
        for label, options in (("fp32", {}), ("int8", {"quantize": 8, "data": str(data.resolve())})):
            path = Path(YOLO(local_model).export(format="litert", imgsz=size, batch=1, device="cpu", **options))
            dest = output / f"wearable_{label}_{size}.tflite"
            # Smoke-test both files with their actual inference backend before publishing.
            inp = next((calibration / "images").glob("*.jpg"))
            YOLO(path, task="detect").predict(str(inp), imgsz=size, device="cpu", verbose=False)
            shutil.copyfile(path, dest)
            artifacts.append({"path": dest.name, "bytes": dest.stat().st_size})
    print(json.dumps(artifacts, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("wearable.pt"))
    parser.add_argument("--calibration", type=Path, default=Path("calibration"))
    parser.add_argument("--output", type=Path, default=Path("exports"))
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    export_pair(args.model, args.calibration, args.output, args.imgsz)
