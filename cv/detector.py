from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None


@dataclass(slots=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]

    @property
    def center(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) // 2, (y1 + y2) // 2)


class DoughMatzahDetector:
    CLASS_NAME_BY_ID: dict[int, str] = {
        0: "Dough Ball",
        1: "Matzah",
    }

    def __init__(self, model_path: str, min_confidence: float = 0.25) -> None:
        if YOLO is None:
            raise RuntimeError("ultralytics is required for detection.")
        self._model = YOLO(model_path)
        self._min_confidence = min_confidence

    def infer(self, frame: np.ndarray) -> list[Detection]:
        results = self._model.predict(
            frame,
            conf=self._min_confidence,
            classes=list(self.CLASS_NAME_BY_ID.keys()),
            verbose=False,
        )
        if not results or results[0].boxes is None:
            return []

        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

        detections: list[Detection] = []
        for box, confidence, class_id in zip(boxes, confs, class_ids):
            if class_id not in self.CLASS_NAME_BY_ID:
                continue
            x1, y1, x2, y2 = map(int, box)
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=self.CLASS_NAME_BY_ID[class_id],
                    confidence=float(confidence),
                    bbox_xyxy=(x1, y1, x2, y2),
                )
            )
        return detections

