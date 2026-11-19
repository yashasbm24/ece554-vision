"""Show wearable detections live. Default to the webcam, or loop a local video."""
import argparse
from pathlib import Path
from time import perf_counter

import cv2
from ultralytics import YOLO

from wearable import CLASSES


def preview(source, model_path):
    camera = source.isdecimal()
    if not camera and not Path(source).is_file():
        raise SystemExit(f"Video file not found: {source}")
    model = YOLO(model_path, task="detect")
    ids = [i for i, label in model.names.items() if label in CLASSES]
    cap = cv2.VideoCapture(int(source) if camera else source)
    title = "ECE 554 | Live detection | Q: quit | Space: pause"
    try:
        if not cap.isOpened():
            raise SystemExit(f"Could not open {source}. For a webcam, close other camera apps or try --source 1.")
        frame_ms = 1000 / max(1, cap.get(cv2.CAP_PROP_FPS) or 25)
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, 960, 540)
        paused = False
        while True:
            start = perf_counter()
            if not paused:
                ok, frame = cap.read()
                if not ok and not camera:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = cap.read()
                if not ok:
                    raise SystemExit(f"No readable frames from {source}.")
                result = model.predict(frame, classes=ids, imgsz=640, conf=0.25,
                                       device="cpu", rect=False, verbose=False)[0]
                cv2.imshow(title, result.plot())
            delay = 30 if paused else 1 if camera else max(1, round(frame_ms - (perf_counter() - start) * 1000))
            key = cv2.waitKey(delay) & 0xff
            if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord(" "):
                paused = not paused
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0", help="Webcam index (default 0) or path to a video")
    args = parser.parse_args()
    preview(args.source, Path(__file__).resolve().parents[1] / "models/yolov8n.pt")
