from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TrackedItem:
    track_id: int
    class_id: int
    center: tuple[int, int]
    bbox_xyxy: tuple[int, int, int, int]
    confidence: float


class CentroidTracker:
    def __init__(self, max_age: int = 20, max_distance: float = 60.0) -> None:
        self._next_id = 1
        self._tracks: dict[int, tuple[int, int]] = {}
        self._ages: dict[int, int] = {}
        self._max_age = max_age
        self._max_distance = max_distance

    def update(self, items: list[TrackedItem]) -> list[TrackedItem]:
        assigned: list[TrackedItem] = []
        used: set[int] = set()
        for item in items:
            track_id = self._best_match(item.center, used)
            if track_id is None:
                track_id = self._next_id
                self._next_id += 1
            used.add(track_id)
            self._tracks[track_id] = item.center
            self._ages[track_id] = 0
            assigned.append(
                TrackedItem(
                    track_id=track_id,
                    class_id=item.class_id,
                    center=item.center,
                    bbox_xyxy=item.bbox_xyxy,
                    confidence=item.confidence,
                )
            )
        self._age_tracks(used)
        return assigned

    def _best_match(self, center: tuple[int, int], used: set[int]) -> int | None:
        best_id: int | None = None
        best_distance = 1e9
        for track_id, prev_center in self._tracks.items():
            if track_id in used:
                continue
            distance = ((center[0] - prev_center[0]) ** 2 + (center[1] - prev_center[1]) ** 2) ** 0.5
            if distance < best_distance and distance <= self._max_distance:
                best_distance = distance
                best_id = track_id
        return best_id

    def _age_tracks(self, used: set[int]) -> None:
        to_delete: list[int] = []
        for track_id in list(self._tracks.keys()):
            if track_id in used:
                continue
            self._ages[track_id] = self._ages.get(track_id, 0) + 1
            if self._ages[track_id] > self._max_age:
                to_delete.append(track_id)
        for track_id in to_delete:
            self._tracks.pop(track_id, None)
            self._ages.pop(track_id, None)

