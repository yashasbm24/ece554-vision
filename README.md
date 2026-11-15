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
