"""Fine-tune on a reproducible, target-enriched sample of COCO train2017."""
import argparse
import random
import zipfile
from pathlib import Path

import yaml
from ultralytics import YOLO


TARGETS = (0, 1, 2, 9, 11, 13, 16, 24, 56)


def prepare(root):
    # Read the compact label archive instead of opening 117k separate files.
    rng = random.Random(554)
    pool = {key: [] for key in TARGETS}
    with zipfile.ZipFile(root / "downloads/coco2017labels.zip") as labels:
        members = sorted(n for n in labels.namelist() if n.startswith("coco/labels/train2017/") and n.endswith(".txt"))
        rng.shuffle(members)
        for name in members:
            classes = {int(line.split()[0]) for line in labels.read(name).splitlines() if line.strip()}
            for key in classes.intersection(pool):
                if len(pool[key]) < 520:
                    pool[key].append(Path(name).stem)
            if all(len(ids) >= 520 for ids in pool.values()):
                break

    folder = root / "datasets/street_sample"
    folder.mkdir(exist_ok=True)
    used = set()
    manifests = {}
    for split, count in (("train", 40), ("dev", 10), ("calibration", 25)):
        chosen = []
        for key in TARGETS:
            available = [i for i in pool[key] if i not in used]
            ids = rng.sample(available, count)
            used.update(ids)
            chosen.extend(ids)
        rng.shuffle(chosen)
        paths = [root / f"datasets/coco/images/train2017/{i}.jpg" for i in chosen]
        if any(not p.is_file() for p in paths):
            raise FileNotFoundError("COCO train2017 images are incomplete")
        manifest = folder / f"{split}.txt"
        manifest.write_text("\n".join(p.as_posix() for p in paths) + "\n", encoding="utf-8")
        manifests[split] = manifest.as_posix()
    names = yaml.safe_load((root / "datasets/coco/data.yaml").read_text(encoding="utf-8-sig"))["names"]
    for name, val in (("train", "dev"), ("calibrate", "calibration")):
        config = dict(path=folder.as_posix(), train=manifests["train"], val=manifests[val], names=names)
        (folder / f"{name}.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return folder / "train.yaml"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=4)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    data = prepare(root)
    model = YOLO(root / "models/yolov8n.pt")
    model.train(data=str(data), epochs=args.epochs, imgsz=640, batch=4, device="cpu", workers=0,
                freeze=10, optimizer="AdamW", lr0=0.0001, warmup_epochs=0, mosaic=0,
                amp=False, seed=554, plots=False, project=str(root / "outputs"), name="street_finetune")
    print(f"Selected checkpoint: {model.trainer.best}")
