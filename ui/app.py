from __future__ import annotations

import time
import tkinter as tk
from tkinter import messagebox, ttk

import cv2

from cv.detector import DoughMatzahDetector, Detection
from ui.calibration import CalibrationController
from ui.table_panel import TablePanel
from ui.video_panel import VideoPanel
from utils.config import AppConfig
from core.zones import find_zone_for_bbox



class BakeryApp:
    def __init__(self, root: tk.Tk, config: AppConfig) -> None:
        self.root = root
        self.config = config
        self.root.title(config.window_title)
        self.root.geometry("1600x880")

        self.calibration = CalibrationController()
        self.source = config.default_source
        self.mode = "roi"
        self._cap: cv2.VideoCapture | None = None
        self._is_playing = False
        self._is_paused = False
        self._frame_index: int = 0
        self._last_detections: list[Detection] = []
        self.zone_states: dict[str, dict] = {}
        self.zone_counters: dict[str, int] = {}

        try:
            self.detector = DoughMatzahDetector(config.model_path, min_confidence=config.min_confidence)
        except Exception as exc:
            self.detector = None
            messagebox.showwarning("Detector Error", f"Detection disabled: {exc}")

        self._build_layout()
        self._bind_events()
        self.root.after(self.config.refresh_interval_ms, self._ui_tick)

    def _build_layout(self) -> None:
        container = ttk.Frame(self.root)
        container.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(container)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right = ttk.Frame(container)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=8, pady=8)

        self.video_panel = VideoPanel(
            left,
            on_play=self.play,
            on_pause=self.pause,
            on_stop=self.stop,
            on_load_source=self.set_source,
            on_close_roi=self.close_roi,
        )
        self.video_panel._on_progress_seek = self._on_progress_seek
        self.video_panel.pack(fill=tk.BOTH, expand=True)
        self.video_panel.source_var.set(self.source)

        calibration_controls = ttk.Frame(left)
        calibration_controls.pack(fill=tk.X, padx=8, pady=4)
        ttk.Button(calibration_controls, text="ROI Mode", command=lambda: self.set_mode("roi")).pack(side=tk.LEFT, padx=3)
        ttk.Button(calibration_controls, text="Worker Zones Mode", command=lambda: self.set_mode("zones")).pack(side=tk.LEFT, padx=3)
        ttk.Button(calibration_controls, text="Clear Calibration", command=self.clear_calibration).pack(side=tk.LEFT, padx=3)

        self.mode_label = ttk.Label(calibration_controls, text="Mode: ROI Calibration")
        self.mode_label.pack(side=tk.RIGHT, padx=15)

        self.table_panel = TablePanel(right)
        self.table_panel.pack(fill=tk.BOTH, expand=True)

    def _bind_events(self) -> None:
        self.video_panel.canvas.bind("<Button-1>", self._on_canvas_click)
        self.video_panel.canvas.bind("<Double-Button-1>", self._on_canvas_double_click)

    def set_source(self, source: str) -> None:
        self.source = source
        self.video_panel.source_var.set(source)
        self.stop()
        self._show_first_frame(source)

    def _show_first_frame(self, source: str) -> None:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            messagebox.showwarning("Source Error", f"Cannot open source: {source}")
            return
        try:
            ok, frame = cap.read()
        finally:
            cap.release()
        if not ok or frame is None:
            messagebox.showwarning("Source Error", f"Cannot read first frame: {source}")
            return
        frame_h, frame_w = frame.shape[:2]
        self.video_panel.set_source_size(frame_w, frame_h)
        self.video_panel.render_frame(frame)
        self.video_panel.draw_calibration(self.calibration.current_roi(), self.calibration.zones)
        self._sync_table_static()

    def play(self) -> None:
        self._sync_table_static()
        if self._cap is None:
            self._cap = cv2.VideoCapture(self.source)
            if not self._cap.isOpened():
                self._cap.release()
                self._cap = None
                messagebox.showwarning("Source Error", f"Cannot open source: {self.source}")
                return
        self._is_playing = True
        self._is_paused = False

    def pause(self) -> None:
        if self._is_playing:
            self._is_paused = True

    def stop(self) -> None:
        self._is_playing = False
        self._is_paused = False
        self._frame_index = 0
        self._last_detections.clear()
        self.zone_states.clear()
        self.zone_counters.clear()
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        text = "ROI Calibration" if mode == "roi" else "Worker Zone Calibration"
        self.mode_label.config(text=f"Mode: {text}")
        if mode == "zones":
            self._sync_table_static()

    def clear_calibration(self) -> None:
        self.calibration.clear_all()
        self.zone_states.clear()
        self.zone_counters.clear()
        self._sync_table_static()
        self.video_panel.draw_calibration(self.calibration.current_roi(), self.calibration.zones)

    def close_roi(self) -> None:
        self.calibration.close_roi()
        if self.calibration.roi_closed:
            self.set_mode("zones")
        self.video_panel.draw_calibration(self.calibration.current_roi(), self.calibration.zones)
        self._sync_table_static()

    def _on_canvas_click(self, event: tk.Event) -> None:
        source_point = self.video_panel.map_display_to_source((int(event.x), int(event.y)))
        self.calibration.on_canvas_click(source_point, self.mode)
        self.video_panel.draw_calibration(self.calibration.current_roi(), self.calibration.zones)
        if self.mode == "zones":
            self._sync_table_static()

    def _on_canvas_double_click(self, event: tk.Event) -> None:
        del event
        self.close_roi()

    def _ui_tick(self) -> None:
        if self._cap is not None and self._is_playing and not self._is_paused:
            ok, frame = self._cap.read()
            if ok and frame is not None:
                infer_every = max(1, int(self.config.infer_every_n_frames))
                should_infer = (self._frame_index % infer_every) == 0
                self._render_with_detections(frame, should_infer=should_infer)
                self._frame_index += 1
                # Update progress display during playback
                self._update_progress_display()
            else:
                self.stop()
        self.root.after(self.config.refresh_interval_ms, self._ui_tick)

    def _render_with_detections(self, frame, should_infer: bool) -> None:
        frame_h, frame_w = frame.shape[:2]
        self.video_panel.set_source_size(frame_w, frame_h)
        rendered = frame.copy()
        if self.detector is not None:
            roi = self.calibration.current_roi()
            if should_infer or not self._last_detections:
                self._last_detections = self.detector.infer(frame)
            detections = self._last_detections
            filtered = [det for det in detections if not roi.is_valid() or roi.contains(det.center)]
            worker_zones = list(self.calibration.zones)
            for det in filtered:
                in_worker_zone = find_zone_for_bbox(worker_zones, det.bbox_xyxy) is not None
                if det.class_id in (0, 1) and in_worker_zone:
                    class_name = DoughMatzahDetector.CLASS_NAME_BY_ID.get(det.class_id, str(det.class_id))
                    self._draw_detection(rendered, det.bbox_xyxy, class_name, det.confidence, in_worker_zone=True)

            # Tracking and counting logic for matzah (one per zone)
            current_time = time.monotonic()
            zones_with_matza = set()
            for det in filtered:
                if det.class_id == 1:  # Matzah
                    zone = find_zone_for_bbox(worker_zones, det.bbox_xyxy)
                    if zone is not None:
                        zones_with_matza.add(zone.worker_id)
                        if zone.worker_id not in self.zone_states:
                            print(f"Starting matzah in zone {zone.worker_id}")
                            self.zone_states[zone.worker_id] = {'last_seen': current_time, 'counted': False}
                        else:
                            print(f"Updating matzah in zone {zone.worker_id}")
                            self.zone_states[zone.worker_id]['last_seen'] = current_time
                            self.zone_states[zone.worker_id]['counted'] = False
            # Check for disappeared matzah in zones
            for zone in self.calibration.zones:
                if zone.worker_id not in zones_with_matza:
                    if zone.worker_id in self.zone_states:
                        elapsed = current_time - self.zone_states[zone.worker_id]['last_seen']
                        print(f"Zone {zone.worker_id} no matzah, elapsed: {elapsed:.2f}s")
                        if elapsed > 3.0 and not self.zone_states[zone.worker_id].get('counted', False):
                            print(f"Counting matzah for zone {zone.worker_id}")
                            self.zone_counters[zone.worker_id] = self.zone_counters.get(zone.worker_id, 0) + 1
                            self.zone_states[zone.worker_id]['counted'] = True
                        elif elapsed <= 3.0:
                            print(f"Zone {zone.worker_id} in grace period")
            # Update table
            self._sync_table_static()

        self.video_panel.render_frame(rendered)
        self.video_panel.draw_calibration(self.calibration.current_roi(), self.calibration.zones)

    def _draw_detection(
        self,
        frame,
        bbox_xyxy: tuple[int, int, int, int],
        class_name: str,
        confidence: float,
        in_worker_zone: bool = False,
    ) -> None:
        x1, y1, x2, y2 = bbox_xyxy
        if class_name == "Dough Ball":
            color = (40, 180, 255)
        elif in_worker_zone:
            # Matzah bbox has >80% overlap with a worker zone.
            color = (40, 40, 255)
        else:
            color = (40, 255, 40)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name} {confidence:.2f}"
        if class_name != "Dough Ball" and in_worker_zone:
            label = f"{label} IN"
        cv2.putText(
            frame,
            label,
            (x1, max(16, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    def _sync_table_static(self) -> None:
        rows = [(zone.worker_id, self.zone_counters.get(zone.worker_id, 0)) for zone in self.calibration.zones]
        self.table_panel.update_rows(rows)

    def _on_progress_seek(self, progress_value: float) -> None:
        """Handle progress bar seek event."""
        if self._cap is not None and self._cap.isOpened():
            # Get total frames and current position
            total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames > 0:
                # Calculate target frame based on progress (0-100)
                target_frame = int((progress_value / 100.0) * total_frames)
                # Set the position
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                self._frame_index = target_frame
                # Update progress display
                self._update_progress_display()
                # Render the frame at the new position
                ok, frame = self._cap.read()
                if ok and frame is not None:
                    self._render_with_detections(frame, should_infer=True)

    def _update_progress_display(self) -> None:
        """Update progress bar and label with current video position."""
        if self._cap is not None and self._cap.isOpened():
            current_frame = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
            total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = self._cap.get(cv2.CAP_PROP_FPS)
            
            if total_frames > 0 and fps > 0:
                # Calculate progress percentage
                progress = (current_frame / total_frames) * 100
                self.video_panel.set_progress(progress)
                
                # Calculate time display
                current_time = current_frame / fps
                total_time = total_frames / fps
                
                current_min = int(current_time // 60)
                current_sec = int(current_time % 60)
                total_min = int(total_time // 60)
                total_sec = int(total_time % 60)
                
                time_text = f"{current_min:02d}:{current_sec:02d} / {total_min:02d}:{total_sec:02d}"
                self.video_panel.set_progress_label(time_text)

