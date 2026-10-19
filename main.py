from __future__ import annotations

import tkinter as tk

from ui.app import BakeryApp
from utils.config import AppConfig


def main() -> None:
    root = tk.Tk()
    config = AppConfig()
    BakeryApp(root, config)
    root.mainloop()


if __name__ == "__main__":
    main()

