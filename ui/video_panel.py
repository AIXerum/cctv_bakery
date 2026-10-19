from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, ttk
from typing import Callable

import cv2
from PIL import Image, ImageTk

from core.zones import PolygonROI, WorkerZone

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720


class VideoPanel(ttk.Frame):
    def __init__(
        self,
        parent: tk.Widget,
        on_play: Callable[[], None],
        on_pause: Callable[[], None],
        on_stop: Callable[[], None],
        on_load_source: Callable[[str], None],
        on_close_roi: Callable[[], None],
    ) -> None:
        super().__init__(parent)
        self._on_play = on_play
        self._on_pause = on_pause
        self._on_stop = on_stop
        self._on_load_source = on_load_source
        self._on_close_roi = on_close_roi
        self._imgtk: ImageTk.PhotoImage | None = None
        self._source_size: tuple[int, int] = (DISPLAY_WIDTH, DISPLAY_HEIGHT)

        controls = ttk.Frame(self)
        controls.pack(fill=tk.X, padx=6, pady=6)

        self.source_var = tk.StringVar()
        ttk.Entry(controls, textvariable=self.source_var, width=50).pack(side=tk.LEFT, padx=2)
        ttk.Button(controls, text="Use Source", command=self._use_text_source).pack(side=tk.LEFT, padx=2)
        ttk.Button(controls, text="Load Video/RTSP", command=self._pick_source).pack(side=tk.LEFT, padx=2)
        ttk.Button(controls, text="Play", command=self._on_play).pack(side=tk.LEFT, padx=2)
        ttk.Button(controls, text="Pause", command=self._on_pause).pack(side=tk.LEFT, padx=2)
        ttk.Button(controls, text="Stop", command=self._on_stop).pack(side=tk.LEFT, padx=2)
        ttk.Button(controls, text="Close ROI", command=self._on_close_roi).pack(side=tk.LEFT, padx=8)

        self.canvas = tk.Canvas(
            self,
            bg="black",
            width=DISPLAY_WIDTH,
            height=DISPLAY_HEIGHT,
            highlightthickness=0,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # Progress bar (seek bar)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Scale(
            self,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.progress_var,
            command=self._on_progress_change,
        )
        self.progress_bar.pack(fill=tk.X, padx=6, pady=2)
        self.progress_label = ttk.Label(self, text="00:00 / 00:00")
        self.progress_label.pack(pady=2)

    def set_progress(self, value: float) -> None:
        """Set progress bar value (0.0 to 100.0)."""
        self.progress_var.set(value)

    def set_progress_range(self, min_val: float, max_val: float) -> None:
        """Set progress bar range."""
        self.progress_bar.config(from_=min_val, to=max_val)

    def set_progress_label(self, text: str) -> None:
        """Set progress label text."""
        self.progress_label.config(text=text)

    def _on_progress_change(self, value: str) -> None:
        """Handle progress bar change."""
        # Convert string value from Scale to float
        try:
            progress_value = float(value)
            # Notify parent about progress change
            if hasattr(self, '_on_progress_seek'):
                self._on_progress_seek(progress_value)
        except ValueError:
            pass

    def _pick_source(self) -> None:
        path = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")],
        )
        if path:
            self.source_var.set(path)
            self._on_load_source(path)

    def _use_text_source(self) -> None:
        source = self.source_var.get().strip()
        if source:
            self._on_load_source(source)

    def render_frame(self, frame_bgr) -> None:
        frame_bgr = cv2.resize(frame_bgr, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_AREA)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        self._imgtk = ImageTk.PhotoImage(image=image)
        self.canvas.delete("video")
        self.canvas.create_image(0, 0, image=self._imgtk, anchor=tk.NW, tags="video")

    def draw_calibration(self, roi: PolygonROI, zones: list[WorkerZone]) -> None:
        self.canvas.delete("calibration")
        source_w, source_h = self._source_size
        if roi.points:
            scaled_roi_points = [self._scale_source_to_display(point, source_w, source_h) for point in roi.points]
            flat_points = [coord for point in scaled_roi_points for coord in point]
            if len(flat_points) >= 4:
                self.canvas.create_line(*flat_points, fill="cyan", width=2, tags="calibration")
            if roi.is_valid():
                self.canvas.create_polygon(*flat_points, outline="cyan", fill="", width=2, tags="calibration")
        for zone in zones:
            x1, y1 = self._scale_source_to_display((zone.x1, zone.y1), source_w, source_h)
            x2, y2 = self._scale_source_to_display((zone.x2, zone.y2), source_w, source_h)
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="yellow", width=2, tags="calibration")
            self.canvas.create_text(
                min(x1, x2) + 5,
                min(y1, y2) - 8,
                text=zone.worker_id,
                anchor=tk.W,
                fill="yellow",
                tags="calibration",
            )

    def set_source_size(self, width: int, height: int) -> None:
        self._source_size = (max(width, 1), max(height, 1))

    def map_display_to_source(self, point: tuple[int, int]) -> tuple[int, int]:
        source_w, source_h = self._source_size
        sx = source_w / DISPLAY_WIDTH
        sy = source_h / DISPLAY_HEIGHT
        x = int(round(point[0] * sx))
        y = int(round(point[1] * sy))
        return (max(0, min(x, source_w - 1)), max(0, min(y, source_h - 1)))

    @staticmethod
    def _scale_source_to_display(
        point: tuple[int, int],
        source_w: int,
        source_h: int,
    ) -> tuple[int, int]:
        dx = DISPLAY_WIDTH / max(source_w, 1)
        dy = DISPLAY_HEIGHT / max(source_h, 1)
        x = int(round(point[0] * dx))
        y = int(round(point[1] * dy))
        return x, y

