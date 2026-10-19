from pathlib import Path

import cv2
from ultralytics import YOLO


def detect_people_video(
    input_video: str = "1.mp4",
    output_video: str = "1_people_detected.mp4",
    model_name: str = "yolov8n.pt",
    conf: float = 0.25,
) -> Path:
    video_path = Path(input_video)
    output_path = Path(output_video)

    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    # Base YOLO model (auto-downloads if missing).
    model = YOLO(model_name)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Class 0 in COCO is "person".
            results = model.predict(frame, conf=conf, classes=[0], verbose=False)
            annotated = results[0].plot()
            writer.write(annotated)
    finally:
        cap.release()
        writer.release()

    return output_path


if __name__ == "__main__":
    saved_path = detect_people_video(
        input_video="1.mp4",
        output_video="1_people_detected.mp4",
        model_name="yolov8n.pt",
        conf=0.25,
    )
    print(f"People detection complete. Saved: {saved_path}")
