from pathlib import Path
import cv2


def split_first_two_minutes(input_video: str, output_video: str, duration_seconds: int = 120) -> None:
    input_path = Path(input_video)
    output_path = Path(output_video)

    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

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

    max_frames = int(fps * duration_seconds)
    written = 0

    try:
        while written < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            written += 1
    finally:
        cap.release()
        writer.release()


if __name__ == "__main__":
    input_file = "Copy of Rolling table 1_2026_02_18_1PM_15_07.mp4"
    output_file = "Copy of Rolling table 1_2026_02_18_1PM_15_07_first_2mins.mp4"

    split_first_two_minutes(input_file, output_file)
    print(f"Saved: {output_file}")
