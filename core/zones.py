from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np


Point = tuple[int, int]


@dataclass(slots=True)
class PolygonROI:
    points: list[Point]

    def is_valid(self) -> bool:
        return len(self.points) >= 3

    def contains(self, point: Point) -> bool:
        if not self.is_valid():
            return True
        contour = np.array(self.points, dtype=np.int32)
        result = cv2.pointPolygonTest(contour, point, False)
        return result >= 0

    def draw(self, frame: np.ndarray) -> None:
        if not self.is_valid():
            return
        contour = np.array(self.points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [contour], isClosed=True, color=(255, 255, 0), thickness=2)


@dataclass(slots=True)
class WorkerZone:
    worker_id: str
    x1: int
    y1: int
    x2: int
    y2: int

    def contains(self, point: Point) -> bool:
        min_x, max_x = sorted((self.x1, self.x2))
        min_y, max_y = sorted((self.y1, self.y2))
        return min_x <= point[0] <= max_x and min_y <= point[1] <= max_y

    def contains_bbox(self, bbox_xyxy: tuple[int, int, int, int]) -> bool:
        """
        Returns True when the IOU between the bbox and this worker zone covers more than 80% of the bbox area.

        bbox_xyxy is expected in source-coordinate space: (x1, y1, x2, y2).
        """
        # Calculate zone bounds
        zx1, zx2 = sorted((self.x1, self.x2))
        zy1, zy2 = sorted((self.y1, self.y2))

        # Calculate bbox bounds
        bx1, by1, bx2, by2 = bbox_xyxy
        bx_min_x, bx_max_x = sorted((bx1, bx2))
        bx_min_y, bx_max_y = sorted((by1, by2))

        # Calculate intersection
        ix1 = max(zx1, bx_min_x)
        iy1 = max(zy1, bx_min_y)
        ix2 = min(zx2, bx_max_x)
        iy2 = min(zy2, bx_max_y)

        if ix1 >= ix2 or iy1 >= iy2:
            return False  # No intersection

        intersection_area = (ix2 - ix1) * (iy2 - iy1)
        bbox_area = (bx_max_x - bx_min_x) * (bx_max_y - bx_min_y)

        if bbox_area == 0:
            return False

        return (intersection_area / bbox_area) > 0.8

    def draw(self, frame: np.ndarray) -> None:
        cv2.rectangle(frame, (self.x1, self.y1), (self.x2, self.y2), (0, 255, 255), 2)
        cv2.putText(
            frame,
            self.worker_id,
            (min(self.x1, self.x2), min(self.y1, self.y2) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )


def find_zone_for_point(zones: Iterable[WorkerZone], point: Point) -> WorkerZone | None:
    for zone in zones:
        if zone.contains(point):
            return zone
    return None


def find_zone_for_bbox(
    zones: Iterable[WorkerZone],
    bbox_xyxy: tuple[int, int, int, int],
) -> WorkerZone | None:
    for zone in zones:
        if zone.contains_bbox(bbox_xyxy):
            return zone
    return None

