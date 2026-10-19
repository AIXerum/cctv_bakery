from pathlib import Path

import cv2
from ultralytics import YOLO


def detect_video(
    weights_path: str = "weights.pt",
    input_video: str = "1.mp4",
    output_video: str = "1_detected.mp4",
    conf: float = 0.25,
) -> Path:
    model_path = Path(weights_path)
    video_path = Path(input_video)
    output_path = Path(output_video)

    if not model_path.exists():
        raise FileNotFoundError(f"Model weight not found: {model_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    model = YOLO(str(model_path))
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

            # Run inference on each frame and draw detections.
            results = model.predict(frame, conf=conf, verbose=False)
            annotated = results[0].plot()
            writer.write(annotated)
    finally:
        cap.release()
        writer.release()

    return output_path


if __name__ == "__main__":
    saved_path = detect_video(
        weights_path="weights.pt",
        input_video="1.mp4",
        output_video="1_detected.mp4",
        conf=0.25,
    )
    print(f"Detection complete. Saved: {saved_path}")
