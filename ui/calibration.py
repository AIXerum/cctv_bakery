from __future__ import annotations

from core.zones import Point, PolygonROI, WorkerZone


class CalibrationController:
    def __init__(self) -> None:
        self.roi_points: list[Point] = []
        self.roi_closed = False
        self.zones: list[WorkerZone] = []
        self._rect_start: Point | None = None

    def on_canvas_click(self, point: Point, mode: str) -> None:
        if mode == "roi" and not self.roi_closed:
            self.roi_points.append(point)
        elif mode == "zones":
            if self._rect_start is None:
                self._rect_start = point
            else:
                worker_id = f"Worker {len(self.zones) + 1}"
                self.zones.append(
                    WorkerZone(worker_id=worker_id, x1=self._rect_start[0], y1=self._rect_start[1], x2=point[0], y2=point[1])
                )
                self._rect_start = None

    def close_roi(self) -> None:
        if len(self.roi_points) >= 3:
            self.roi_closed = True

    def clear_all(self) -> None:
        self.roi_points.clear()
        self.roi_closed = False
        self.zones.clear()
        self._rect_start = None

    def current_roi(self) -> PolygonROI:
        return PolygonROI(points=list(self.roi_points))

