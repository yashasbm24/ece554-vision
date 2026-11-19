"""Bundle a chosen checkpoint and the 225 calibration samples for LiteRT export."""
import argparse
import zipfile
from pathlib import Path


def pack(model, destination):
    root = Path(__file__).resolve().parents[1]
    samples = (root / "datasets/street_sample/calibration.txt").read_text().splitlines()
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(model, "wearable.pt")
        for entry in samples:
            image = Path(entry)
            bundle.write(image, f"calibration/images/{image.name}")
            label = root / "datasets/coco/labels/train2017" / (image.stem + ".txt")
            bundle.write(label, f"calibration/labels/{label.name}")
        for tool in ("export_wearable.py", "wearable.py", "evaluate_model.py"):
            bundle.write(root / "src" / tool, tool)
        bundle.write(root / "inputs/videos/coco_photo_replay.mp4", "replay.mp4")
        for wav in (root / "models/alerts").glob("*.wav"):
            bundle.write(wav, f"models/alerts/{wav.name}")
        bundle.writestr("README.txt", f"""Source checkpoint: {model.name} ({model.parent.name})

Extract this archive on x86-64 Linux. From its directory in a Python 3.12 environment:
python -m pip install "ultralytics[export-litert]==8.4.212"
python export_wearable.py --imgsz 640
python export_wearable.py --imgsz 320

The default export uses wearable.pt and the 225 COCO train2017 calibration JPEGs here.
Each run writes two .tflite files in exports/ and tries one prediction with each.
For accuracy, use evaluate_model.py with each model and the same labeled COCO val2017
dataset used for the baseline. Set the path in the dataset YAML for the evaluation host.

Copy the exports, wearable.py, replay.mp4 and models/alerts to the Pi.
Install the project version of ultralytics, ai-edge-litert and OpenCV on a compatible 64-bit Pi OS.
From the copied directory run each model on the identical replay, for example:
python wearable.py --model exports/wearable_fp32_640.tflite --source replay.mp4 --output fp32_640
python wearable.py --model exports/wearable_int8_640.tflite --source replay.mp4 --output int8_640
For the 320 models add --imgsz 320 and use a fresh output directory for each run.
benchmark.json records median model and capture-to-detections times after warmup.
The replay contains COCO still images. For a live USB camera, use --source 0.
Use --speak to announce a central object. Distance-based urgency requires sensor integration.
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pack(args.model, args.output)
