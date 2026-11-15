"""Evaluate a YOLO detector and export per-class metrics."""
import argparse
import csv
import json
from pathlib import Path

import yaml
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/yolov8n.pt")
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--name", default="evaluation")
    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    config = yaml.safe_load(data_path.read_text(encoding="utf-8-sig"))
    model = YOLO(args.model)
    names = config["names"]
    dataset_names = dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}
    if dataset_names != model.names:
        raise ValueError(
            "Dataset class IDs/names differ from the model. Map labels to the "
            "model's classes or evaluate a checkpoint trained for this dataset."
        )

    results = model.val(
        data=str(data_path), split=args.split, imgsz=args.imgsz,
        batch=1, device=args.device, workers=0, plots=True,
        conf=0.001, project=str(Path("outputs").resolve()), name=args.name,
    )
    output = Path(results.save_dir)
    rows = []
    for position, class_id in enumerate(results.box.ap_class_index):
        precision = float(results.box.p[position])
        recall = float(results.box.r[position])
        rows.append({
            "class_id": int(class_id),
            "class": model.names[int(class_id)],
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "AP50": float(results.box.ap50[position]),
            "AP50_95": float(results.box.ap[position]),
        })
    if not rows:
        raise RuntimeError("No evaluable classes. Check the dataset labels.")
    with (output / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "model": args.model,
        "dataset": str(data_path),
        "split": args.split,
        "imgsz": args.imgsz,
        "precision_macro": float(results.box.mp),
        "recall_macro": float(results.box.mr),
        "f1_macro": sum(row["f1"] for row in rows) / len(rows),
        "mAP50": float(results.box.map50),
        "mAP50_95": float(results.box.map),
        "speed_ms_per_image": {k: float(v) for k, v in results.speed.items()},
        "f1_definition": (
            "Mean of per-class F1 computed from Ultralytics validation P/R at "
            "its selected best-F1 operating point, IoU 0.5. Confidence floor "
            "0.001 retains detections for AP curves. This is validation-selected "
            "F1, not a separately fixed deployment-threshold F1."
        ),
    }
    (output / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved metrics and plots in {output}")


if __name__ == "__main__":
    main()
