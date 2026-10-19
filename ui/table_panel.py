from __future__ import annotations

import csv
import tkinter as tk
from tkinter import ttk, messagebox, filedialog


class TablePanel(ttk.Frame):
    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent)
        self.tree = ttk.Treeview(
            self,
            columns=("zone_id", "completed_matzah"),
            show="headings",
            height=18,
        )
        self.tree.heading("zone_id", text="Zone ID")
        self.tree.heading("completed_matzah", text="Completed Matzah")
        self.tree.column("zone_id", width=140, anchor=tk.W)
        self.tree.column("completed_matzah", width=160, anchor=tk.CENTER)
        self.tree.pack(fill=tk.BOTH, expand=True)

        # Add Save button
        self.save_button = ttk.Button(self, text="Save to CSV", command=self.save_to_csv)
        self.save_button.pack(pady=5)

    def update_rows(self, rows: list[tuple[str, int]]) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)
        for zone_id, count in rows:
            self.tree.insert("", tk.END, values=(zone_id, count))

    def save_to_csv(self) -> None:
        """Save the current table data to a CSV file."""
        if not self.tree.get_children():
            messagebox.showwarning("No Data", "There is no data to save.")
            return

        # Ask user for file location
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Save CSV File"
        )
        
        if not file_path:
            return  # User cancelled

        try:
            with open(file_path, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                # Write header
                writer.writerow(["Zone ID", "Completed Matzah"])
                # Write data
                for item in self.tree.get_children():
                    values = self.tree.item(item, 'values')
                    writer.writerow(values)
            
            messagebox.showinfo("Success", f"Data successfully saved to {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file: {str(e)}")

