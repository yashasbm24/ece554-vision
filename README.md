# ECE 554 Vision

YOLO object-detection evaluation for ECE 554. The evaluation script exports
per-class precision, recall, F1, AP50, and AP50-95, along with aggregate metrics
and validation plots.

## Setup

Create a Python environment and install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Dependencies are pinned to the versions present in the original project environment.
Install the appropriate PyTorch build separately if GPU support is required.

## Data and weights

Download the COCO images and corresponding YOLO-format labels separately.
Dataset configurations are included in [datasets/coco](datasets/coco/data.yaml)
and [datasets/coco2017-val-subset](datasets/coco2017-val-subset/data.yaml).
Edit the `path` field in the selected configuration to point to your local
dataset directory; the checked-in values describe the original workstation.
Keep images under `images/val2017` and corresponding labels under `labels/val2017`.
The subset configuration requires the matching subset data, which is not bundled.

Place the pretrained YOLOv8n weights at `models/yolov8n.pt`, or pass
`--model yolov8n.pt` to let Ultralytics download the standard model when needed.
Dataset class IDs and names must match those of the model.

## Evaluation

Run from the repository root:

```powershell
python src/evaluate_model.py --model yolov8n.pt --data datasets/coco/data.yaml --device cpu --name coco_baseline
```

Use `--device 0` for the first CUDA GPU when available. Additional options include
`--split val`, `--imgsz 640`, and `--name evaluation`.

Results are saved under `outputs/<name>/`, including `per_class_metrics.csv`,
`metrics.json`, and plots. Ultralytics may increment the directory name for
repeated runs. F1 uses the validation-selected operating point; it is not a
separately fixed deployment-threshold metric.

The Python environment, datasets, downloads, inputs, model weights, and generated
outputs are kept locally and excluded from Git, except for the dataset
configuration files and subset documentation.

## Wearable experiment

The CPU fine-tune uses 360 target-enriched COCO train2017 images, 90 separate
images from train2017 for checkpoint selection, and a further 225 for calibration.
It preserves the original 80 class IDs. The runtime filters predictions to the nine
wearable classes. Keep the 5,000 val2017 images for reporting.

```powershell
.\ece554\Scripts\python.exe src/finetune_street.py
.\ece554\Scripts\python.exe src/evaluate_model.py --model outputs/street_finetune/weights/best.pt --data datasets/coco/data.yaml --name coco_after_finetuning
```

Run the detector on a video, or pass `--source 0` for a USB camera. Give each
run a fresh output directory. Add `--speak` to announce objects in the center
of the frame. The nine recorded WAV files are in `models/alerts`.

```powershell
.\ece554\Scripts\python.exe src/wearable.py --model models/yolov8n.pt --source inputs/videos/coco_photo_replay.mp4 --output outputs/replay_demo --save-video
```

Each line of `detections.jsonl` contains a frame index, dimensions, capture
timestamp, and a list of class names, confidence scores, and pixel coordinates
in left, top, right, bottom order. The distance sensor supplies the range needed
to determine alert urgency. The COCO replay consists of nine photos held for
ten frames each.

For LiteRT export, `src/pack_export.py --model PATH --output KIT.zip` packages
the chosen checkpoint, the calibration images and labels, the replay, audio,
and the export and benchmark scripts. The included README gives the FP32 and
INT8 commands for x86-64 Linux and the benchmarking commands for the Pi.
