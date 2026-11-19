"""Run the nine-class detector on a video or camera and publish frame detections."""
import argparse
import json
import platform
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from time import monotonic, perf_counter

import cv2
from ultralytics import YOLO


CLASSES = {"person", "chair", "bicycle", "car", "dog", "bench", "backpack", "traffic light", "stop sign"}


def speech_command(label, directory):
    clip = directory / (label.replace(" ", "_") + ".wav")
    if sys.platform == "win32":
        if not clip.is_file():
            raise FileNotFoundError(f"Missing speech clip: {clip}")
        return [sys.executable, "-c", "import sys,winsound; winsound.PlaySound(sys.argv[1],winsound.SND_FILENAME | winsound.SND_NODEFAULT)", str(clip)]
    if clip.is_file() and shutil.which("aplay"):
        return ["aplay", "-q", str(clip)]
    engine = shutil.which("espeak-ng") or shutil.which("espeak")
    if not engine:
        raise RuntimeError("Install espeak-ng to use --speak.")
    return [engine, label]


def announce_candidate(detections, width):
    # The team's distance/urgency logic consumes the full detection list.
    candidates = [d for d in detections if d["box"][0] <= width / 2 <= d["box"][2]]
    return max(candidates, key=lambda d: d["confidence"], default=None)


def run(args):
    model = YOLO(args.model, task="detect")
    class_ids = [i for i, label in model.names.items() if label in CLASSES]
    if len(class_ids) != len(CLASSES):
        raise ValueError("All nine wearable classes must exist in the checkpoint.")
    capture = cv2.VideoCapture(int(args.source) if args.source.isdecimal() else args.source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source {args.source}")
    writer = speaker = None
    frames = 0
    frame_times, model_times = [], []
    next_speech = 0
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        with (args.output / "detections.jsonl").open("x", encoding="utf-8") as records:
            while args.frames is None or frames < args.frames:
                start = perf_counter()
                ok, frame = capture.read()
                if not ok:
                    break
                result = model.predict(frame, classes=class_ids, imgsz=args.imgsz, conf=args.conf,
                                       device="cpu", rect=False, verbose=False)[0]
                detections = [{"class": model.names[int(b[5])], "confidence": float(b[4]), "box": b[:4]}
                              for b in result.boxes.data.cpu().tolist()]
                height, width = frame.shape[:2]
                record = dict(frame=frames, capture_time_ms=capture.get(cv2.CAP_PROP_POS_MSEC),
                              width=width, height=height, detections=detections)
                if args.speak:
                    candidate = announce_candidate(detections, width)
                    if speaker is not None and speaker.poll() not in (None, 0):
                        raise RuntimeError("Speech engine failed")
                    if candidate and monotonic() >= next_speech and (speaker is None or speaker.poll() is not None):
                        label = candidate["class"]
                        speaker = subprocess.Popen(speech_command(label, args.audio_dir))
                        next_speech = monotonic() + 3
                if frames >= args.warmup:
                    model_times.append(result.speed["inference"])
                    frame_times.append((perf_counter() - start) * 1000)
                records.write(json.dumps(record) + "\n")
                if args.save_video:
                    if writer is None:
                        writer = cv2.VideoWriter(str(args.output / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                                 capture.get(cv2.CAP_PROP_FPS) or 15, (width, height))
                        if not writer.isOpened():
                            raise RuntimeError("Could not create annotated.mp4")
                    writer.write(result.plot())
                frames += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if speaker is not None:
            if speaker.wait() != 0:
                raise RuntimeError("Speech engine failed")
    if not frame_times:
        raise RuntimeError("No frames measured after warmup. Check the source or reduce --warmup.")
    report = dict(model=str(args.model), model_bytes=Path(args.model).stat().st_size,
                  source=args.source, platform=platform.platform(), machine=platform.machine(),
                  input_size=args.imgsz, confidence=args.conf, frames=frames, warmup=args.warmup,
                  measured_frames=len(frame_times), median_inference_ms=statistics.median(model_times),
                  median_capture_to_detections_ms=statistics.median(frame_times),
                  scope="Includes capture/decode, prediction and preparing detections. Excludes logging, rendering and physical alert latency.")
    (args.output / "benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/yolov8n.pt")
    parser.add_argument("--source", required=True, help="Video filename or camera index such as 0")
    parser.add_argument("--output", required=True, type=Path, help="New output directory")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--speak", action="store_true")
    parser.add_argument("--audio-dir", type=Path, default=Path("models/alerts"))
    run(parser.parse_args())
