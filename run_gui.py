# Kasha
# Copyright (C) 2025 Samuel Markovich
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
import os
import sys
import threading
from video_classifier.core import main as run_classifier_logic, generate_proxies
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter
from typing import Any, Dict

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

        # A simple class to mimic the argparse Namespace object that main() expects
class Args:
    def __init__(self, d):
        self.__dict__.update(d)

# ----------------------------- Dynamic Paths -----------------------------
HOME = os.path.expanduser("~")
# The root directory of the project is the parent of the folder containing this script.
# In our new structure, this script (run_gui.py) is in the root, so its parent is the root.
# Using .parent.resolve() makes it robust if you ever move the script.
PROJECT_ROOT = Path(__file__).parent.resolve()

# Build the path to the config file relative to the project root
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.xml"

# Extensions used in Preflight
VIDEO_EXTS = (".mp4", ".mov", ".m4v", ".avi", ".mts", ".mxf", ".mkv", ".wmv", ".mpg", ".mpeg", ".3gp", ".m2ts")

# ----------------------------- Tooltip helper -----------------------------
class Tooltip:
    """
    Lightweight tooltip: appears after a short delay when hovering on a widget.
    """
    def __init__(self, widget, text: str, delay_ms: int = 400, wraplength=380):
        self.widget = widget
        self.text = text
        self.delay = delay_ms
        self.wraplength = wraplength
        self._id = None
        self.tip = None
        widget.bind("<Enter>", self._schedule)
        widget.bind("<Leave>", self._hide)
        widget.bind("<ButtonPress>", self._hide)

    def _schedule(self, _event=None):
        self._cancel()
        self._id = self.widget.after(self.delay, self._show)

    def _cancel(self):
        if self._id:
            try:
                self.widget.after_cancel(self._id)
            except Exception:
                pass
            self._id = None

    def _show(self):
        if self.tip or not self.text:
            return
        x, y, cx, cy = self.widget.bbox("insert") if hasattr(self.widget, "bbox") else (0, 0, 0, 0)
        x = x + self.widget.winfo_rootx() + 20
        y = y + self.widget.winfo_rooty() + cy + 20
        self.tip = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw, text=self.text, justify="left",
            relief="solid", borderwidth=1,
            background="#ffffe0",
            foreground="black",  # <-- Add this line
            wraplength=self.wraplength
        )
        label.pack(ipadx=8, ipady=5)

    def _hide(self, _event=None):
        self._cancel()
        if self.tip:
            try:
                self.tip.destroy()
            except Exception:
                pass
        self.tip = None

# ----------------------------- Config Model -----------------------------
XML_FIELDS = [
    ("root", str),
    ("output", str),
    ("interval", float),
    ("max_frames", int),
    ("vad_window", float),
    ("vad_thresh", float),
    ("vad_min_speech", float),
    ("vad_max_gap", float),
    ("workers", int),
    ("short_max", float),
    ("long_min", float),
    ("min_duration", float),
    ("session_scope", str),
    ("shoot_depth", int),
    ("session_overlap", float),
    ("session_gap_tol", int),
    ("dedupe", str),
    ("log_img_thresh", float),
    ("enable_exiftool", str),

    # interview/dedupe thresholds (script supports these)
    ("min_duration_sec", float),
    ("min_speech_ratio", float),
    ("min_longest_speech_sec", float),
    ("min_segments_count", int),
    ("vad_aggressiveness", int),
    ("duration_tolerance_sec", float),
]

DEFAULTS: Dict[str, Any] = {
    "root": "",
    "output": "",
    # legacy / placeholders kept to not break anything
    "interval": 2.0,
    "max_frames": 400,
    "vad_window": 0.5,
    "vad_thresh": 0.5,
    "vad_min_speech": 0.3,
    "vad_max_gap": 0.5,
    "workers": 4,
    "short_max": 5.0,
    "long_min": 8.0,
    "min_duration": 0.5,
    "session_scope": "shoot",
    "shoot_depth": 2,
    "session_overlap": 0.30,
    "session_gap_tol": 15,
    "dedupe": "true",
    "log_img_thresh": 0.55,
    "enable_exiftool": "false",

    # interview detector defaults (match script DEFAULTS)
    "min_duration_sec": 60.0,
    "min_speech_ratio": 0.15,
    "min_longest_speech_sec": 10.0,
    "min_segments_count": 2,
    "vad_aggressiveness": 1,
    "duration_tolerance_sec": 3.0,
}

# Beginner-friendly preset mappings
SENSITIVITY_PRESETS = {
    # lenient → more likely to flag as interview
    "Lenient (find more)": dict(min_duration_sec=60.0, min_speech_ratio=0.15, min_longest_speech_sec=10.0, min_segments_count=2, vad_aggressiveness=1),
    # balanced → default
    "Balanced (recommended)": dict(min_duration_sec=120.0, min_speech_ratio=0.25, min_longest_speech_sec=20.0, min_segments_count=3, vad_aggressiveness=2),
    # strict → fewer false positives
    "Strict (fewer false hits)": dict(min_duration_sec=180.0, min_speech_ratio=0.35, min_longest_speech_sec=30.0, min_segments_count=4, vad_aggressiveness=3),
}

DEDUP_PRESETS = {
    "Off": dict(dedupe="false", session_scope="global", duration_tolerance_sec=3.0),
    "Within shoot (default)": dict(dedupe="true", session_scope="shoot", duration_tolerance_sec=3.0),
    "Across project (global)": dict(dedupe="true", session_scope="global", duration_tolerance_sec=3.0),
}

# Tooltips text
TOOLTIPS = {
    "root": "Select the main folder containing all your video footage. The application will scan all subfolders within this location.",
    "output": "Choose where to save the final CSV report. Two other files, a summary.txt and a list of interviews, will be saved next to it.",
    "sensitivity": "Controls how strictly the app identifies interviews based on speech. 'Lenient' finds more potential interviews, while 'Strict' reduces false positives.",
    "dedupe_mode": "Handles duplicate clips. 'Within shoot' finds duplicates in the same project folder. 'Across project' finds duplicates everywhere. 'Off' disables this.",
    "speed": "Sets the number of CPU cores to use for processing. Higher numbers are faster but use more of your computer's resources.",
    "enable_exiftool": "If checked, the app uses 'exiftool' to read detailed camera metadata (model, creation time). Requires exiftool to be installed on your system.",
    "advanced": "Toggles the visibility of expert-level controls for fine-tuning the classification algorithm.",

    # Advanced fields
    "min_duration_sec": "An interview must be at least this many seconds long.",
    "min_speech_ratio": "Sets the minimum percentage of the clip that must contain speech (e.g., 0.25 = 25% speech).",
    "min_longest_speech_sec": "An interview must contain at least one unbroken block of speech that is this many seconds long.",
    "min_segments_count": "The minimum number of separate, distinct speech segments an interview must have.",
    "vad_aggressiveness": "How strictly the voice detector identifies speech. 0 is very lenient (might count noise as speech), 3 is very strict (might miss quiet speech).",
    "duration_tolerance_sec": "How close in duration (in seconds) two clips must be to be considered potential duplicates.",
    "session_scope": "Defines a 'shoot'. 'shoot' groups files by their parent folder. 'global' treats all files as part of one large project.",
    "shoot_depth": "How many parent folders to use for the 'Project' name. Example: For a path '.../MyProject/Day1/CAM1', a depth of 2 would result in a project name of 'Day1/CAM1'.",
}

# ----------------------------- XML helpers -----------------------------
def load_xml(path: str) -> Dict[str, Any]:
    tree = ET.parse(path)
    root = tree.getroot()
    cfg: Dict[str, Any] = {}
    for key, _typ in XML_FIELDS:
        node = root.find(key)
        if node is not None and node.text is not None:
            cfg[key] = node.text.strip()
    # normalize to strings; we’ll coerce later when building args
    return cfg

def save_xml(path: str, data: Dict[str, Any]) -> None:
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
    root = ET.Element("config")
    for key, _typ in XML_FIELDS:
        el = ET.SubElement(root, key)
        el.text = str(data.get(key, DEFAULTS.get(key, "")))
    tree = ET.ElementTree(root)
    try:
        ET.indent(tree, space="  ")
    except Exception:
        pass
    tree.write(path, encoding="utf-8", xml_declaration=True)

# ----------------------------- GUI App -----------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Video Classifier (Beginner Mode)")
        self.geometry("1000x980")
        self.minsize(980, 900)

        self.config_path = DEFAULT_CONFIG_PATH
        self.vars: Dict[str, tk.Variable] = {k: tk.StringVar(value=str(v)) for k, v in DEFAULTS.items()}

        # Beginner presets state
        self.sensitivity_choice = tk.StringVar(value="Lenient (find more)")
        self.dedupe_choice = tk.StringVar(value="Within shoot (default)")
        self.speed_choice = tk.StringVar(value="Balanced")
        self.advanced_visible = tk.BooleanVar(value=False)

        self._build_ui()

        self.proc: subprocess.Popen | None = None
        self.results_data: list[dict] = []

        self.after(50, self.autoload_config)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def choose_proxy_destination(self):
        """Opens a dialog to choose the proxy destination folder."""
        folder = filedialog.askdirectory(title="Select Proxy Destination Folder")
        if folder:
            self.proxy_dest_var.set(folder)

    def select_all(self):
        """Selects all items in the TreeView."""
        for item in self.tree.get_children():
            self.tree.selection_add(item)

    def deselect_all(self):
        """Deselects all items in the TreeView."""
        for item in self.tree.get_children():
            self.tree.selection_remove(item)

    def on_generate_proxies(self):
    """Gets selected files and runs the proxy generation in a background thread."""
    selected_items = self.tree.selection()
    if not selected_items:
        messagebox.showwarning("No Selection", "Please select at least one file to generate proxies.")
        return

    destination = self.proxy_dest_var.get()
    if not destination or not os.path.isdir(destination):
        messagebox.showwarning("Invalid Destination", "Please select a valid destination folder for the proxies.")
        return

    # Get the full file paths from the selected items in the tree
    file_paths = [self.tree.item(item_id, "text") for item_id in selected_items]

    def progress_callback(current, total, message, is_error=False):
        """Updates the GUI from the background thread."""
        self.progress_var.set((current / total) * 100)
        self.progress_label_var.set(f"({current}/{total}) {message}")
        if is_error:
            self.log(f"[ERROR] {message}")

    def task():
        """The actual work to be done in the background."""
        # Show progress bar and label
        self.progress_bar.pack(side="right", fill="x", expand=True, padx=4)
        self.progress_label.pack(side="right")

        generate_proxies(file_paths, destination, progress_callback)

        # Hide progress bar and label when done
        self.progress_bar.pack_forget()
        self.progress_label.pack_forget()
        self.log("Proxy generation complete.")

    # Run the proxy generation in a separate thread
    threading.Thread(target=task, daemon=True).start()
        
    def populate_results_tree(self):
        """Clears and repopulates the TreeView with the latest results."""
        # Clear any old results from the tree
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Populate the tree with new results
        for result_item in self.results_data:
            # We only want to display the filename, not the full path
            filename = os.path.basename(result_item.get("path", ""))
            classification = result_item.get("class_label", "N/A")
            project = result_item.get("project", "N/A")

            # The 'values' are what's displayed. The 'text' is a hidden identifier.
            # We will store the full path in the 'text' field to retrieve later.
            self.tree.insert(
                "", "end", text=result_item.get("path", ""),
                values=(filename, classification, project)
            )
    def on_tree_click(self, event):
        """Handles double-clicks on the tree to open the file's folder."""
        item_id = self.tree.identify_row(event.y)
        if not item_id:
            return

        # Retrieve the full path we stored in the 'text' field
        file_path = self.tree.item(item_id, "text")
        if file_path and os.path.exists(file_path):
            self.log(f"Opening folder for: {os.path.basename(file_path)}")
            # This command opens the containing folder and selects the file in Finder
            subprocess.run(["open", "-R", file_path])

        
# ------------------- UI -------------------
    def _build_ui(self):
        pad = {"padx": 8, "pady": 6}

        # Beginner panel
        frame_easy = ttk.LabelFrame(self, text="Beginner Settings")
        frame_easy.pack(fill="x", **pad)

        # Root / Output
        row = 0
        ttk.Label(frame_easy, text="Root folder").grid(row=row, column=0, sticky="w")
        e_root = ttk.Entry(frame_easy, textvariable=self.vars["root"], width=65)
        e_root.grid(row=row, column=1, sticky="we", padx=4)
        b_root = ttk.Button(frame_easy, text="Choose Folder", command=self.choose_root)
        b_root.grid(row=row, column=2, padx=4, sticky="e")
        Tooltip(e_root, TOOLTIPS["root"])

        row += 1
        ttk.Label(frame_easy, text="Output CSV").grid(row=row, column=0, sticky="w")
        e_out = ttk.Entry(frame_easy, textvariable=self.vars["output"], width=65)
        e_out.grid(row=row, column=1, sticky="we", padx=4)
        b_out = ttk.Button(frame_easy, text="Choose CSV", command=self.choose_output)
        b_out.grid(row=row, column=2, padx=4, sticky="e")
        Tooltip(e_out, TOOLTIPS["output"])

        # Presets row
        row += 1
        ttk.Label(frame_easy, text="Interview sensitivity").grid(row=row, column=0, sticky="w")
        cmb_sense = ttk.Combobox(frame_easy, values=list(SENSITIVITY_PRESETS.keys()),
                                 textvariable=self.sensitivity_choice, state="readonly", width=28)
        cmb_sense.grid(row=row, column=1, sticky="w", padx=4)
        Tooltip(cmb_sense, TOOLTIPS["sensitivity"])

        row += 1
        ttk.Label(frame_easy, text="Duplicate handling").grid(row=row, column=0, sticky="w")
        cmb_dedupe = ttk.Combobox(frame_easy, values=list(DEDUP_PRESETS.keys()),
                                  textvariable=self.dedupe_choice, state="readonly", width=28)
        cmb_dedupe.grid(row=row, column=1, sticky="w", padx=4)
        Tooltip(cmb_dedupe, TOOLTIPS["dedupe_mode"])

        row += 1
        ttk.Label(frame_easy, text="Performance (CPU workers)").grid(row=row, column=0, sticky="w")
        # We now offer a direct choice of worker counts
        cmb_speed = ttk.Combobox(frame_easy, values=[1, 2, 4, 6, 8],
                                 textvariable=self.vars["workers"], state="readonly", width=28)
        cmb_speed.grid(row=row, column=1, sticky="w", padx=4)
        Tooltip(cmb_speed, TOOLTIPS["speed"])

        row += 1
        chk_exif = ttk.Checkbutton(frame_easy, text="Enrich with ExifTool (camera make/model, timestamps)",
                                   variable=self.vars["enable_exiftool"], onvalue="true", offvalue="false")
        chk_exif.grid(row=row, column=1, sticky="w", padx=4)
        Tooltip(chk_exif, TOOLTIPS["enable_exiftool"])

        frame_easy.grid_columnconfigure(1, weight=1)

        # Preflight + Advanced toggle
        bar = ttk.Frame(self)
        bar.pack(fill="x", **pad)
        ttk.Button(bar, text="Preflight: Count Videos", command=self.on_preflight).pack(side="left")
        chk_adv = ttk.Checkbutton(bar, text="Show Advanced Controls", variable=self.advanced_visible, command=self._toggle_advanced)
        chk_adv.pack(side="right")
        Tooltip(chk_adv, TOOLTIPS["advanced"])

        # Advanced panel (hidden by default)
        self.frame_adv = ttk.LabelFrame(self, text="Advanced Controls (Experts)")
        # Build advanced grid now; show/hide later
        self._build_advanced(self.frame_adv)

               # --- Run and Results ---
        frame_run = ttk.LabelFrame(self, text="Run")
        frame_run.pack(fill="both", expand=True, **pad)

        # --- Top bar for buttons ---
        run_bar = ttk.Frame(frame_run)
        run_bar.pack(fill="x", padx=6, pady=6)

        ttk.Button(run_bar, text="Run Classifier", command=self.on_run).pack(side="left")
        ttk.Button(run_bar, text="Stop", command=self.on_stop).pack(side="left", padx=4)
        self.results_button = ttk.Button(run_bar, text="Show Export Tools", command=self.toggle_results_view)
        self.results_button.pack(side="right")
        self.results_button.state(['disabled']) # Disabled until there are results

        # --- Console ---
        self.console = tk.Text(frame_run, height=10, wrap="word") # Reduced height
        self.console.pack(fill="both", expand=True, padx=6, pady=(0,6))

        # --- Collapsible Results Frame (Initially Hidden) ---
        self.frame_results = ttk.LabelFrame(frame_run, text="Export Tools")
        # We will build the contents of this frame in the next step
        # --- UI elements inside the results frame ---

        # --- Top bar for proxy destination ---
        proxy_bar = ttk.Frame(self.frame_results)
        proxy_bar.pack(fill="x", padx=6, pady=(6,0))

        ttk.Label(proxy_bar, text="Proxy Destination:").pack(side="left")
        self.proxy_dest_var = tk.StringVar()
        proxy_entry = ttk.Entry(proxy_bar, textvariable=self.proxy_dest_var)
        proxy_entry.pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(proxy_bar, text="Choose Folder...", command=self.choose_proxy_destination).pack(side="left")

        # --- TreeView for results ---
        tree_frame = ttk.Frame(self.frame_results)
        tree_frame.pack(fill="both", expand=True, padx=6, pady=6)

        self.tree = ttk.Treeview(tree_frame, columns=("File", "Classification", "Project"), show="headings")
        self.tree.bind("<Double-1>", self.on_tree_click)
        self.tree.heading("File", text="File")
        self.tree.heading("Classification", text="Classification")
        self.tree.heading("Project", text="Project")
        self.tree.column("File", width=400)
        self.tree.column("Classification", width=100)
        self.tree.column("Project", width=200)

        # Add a scrollbar
        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.tree.pack(side="left", fill="both", expand=True)

        # --- Bottom bar for action buttons ---
        action_bar = ttk.Frame(self.frame_results)
        action_bar.pack(fill="x", padx=6, pady=(0,6))
        
        # Progress bar (initially hidden)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(action_bar, variable=self.progress_var, maximum=100)
        # The label for progress text
        self.progress_label_var = tk.StringVar()
        self.progress_label = ttk.Label(action_bar, textvariable=self.progress_label_var)

        ttk.Button(action_bar, text="Select All", command=self.select_all).pack(side="left")
        ttk.Button(action_bar, text="Deselect All", command=self.deselect_all).pack(side="left", padx=4)
        ttk.Button(action_bar, text="Generate Proxies for Selected", command=self.on_generate_proxies).pack(side="right")

        self.rowconfigure(4, weight=1)
        self.columnconfigure(0, weight=1)

    def _build_advanced(self, frame_adv: ttk.LabelFrame):
        grid = ttk.Frame(frame_adv)
        grid.pack(fill="both", expand=True, padx=6, pady=6)

        # rows of advanced controls
        adv_keys = [
            ("min_duration_sec", "Minimum duration (sec)", TOOLTIPS["min_duration_sec"]),
            ("min_speech_ratio", "Min speech ratio (0..1)", TOOLTIPS["min_speech_ratio"]),
            ("min_longest_speech_sec", "Min longest speech (sec)", TOOLTIPS["min_longest_speech_sec"]),
            ("min_segments_count", "Min speech segments", TOOLTIPS["min_segments_count"]),
            ("vad_aggressiveness", "VAD aggressiveness (0-3)", TOOLTIPS["vad_aggressiveness"]),
            ("duration_tolerance_sec", "Duplicate duration tolerance (sec)", TOOLTIPS["duration_tolerance_sec"]),
            ("session_scope", "Session scope", TOOLTIPS["session_scope"]),
            ("shoot_depth", "Shoot folder depth", TOOLTIPS["shoot_depth"]),
            ("workers", "Parallel workers", "How many files to process at once."),
            ("dedupe", "Dedupe", "true/false to turn duplicate removal on/off."),
        ]

        row = 0
        for key, label, tip in adv_keys:
            ttk.Label(grid, text=label).grid(row=row, column=0, sticky="w")
            if key in ("session_scope", "dedupe"):
                if key == "session_scope":
                    w = ttk.Combobox(grid, values=["shoot", "global"], textvariable=self.vars[key], state="readonly", width=14)
                else:
                    w = ttk.Combobox(grid, values=["true", "false"], textvariable=self.vars[key], state="readonly", width=14)
            else:
                w = ttk.Entry(grid, textvariable=self.vars[key], width=20)
            w.grid(row=row, column=1, sticky="w", padx=4, pady=2)
            Tooltip(w, tip)
            row += 1

        grid.grid_columnconfigure(1, weight=1)
        
    def toggle_results_view(self):
        """Shows or hides the results frame."""
        if self.frame_results.winfo_viewable():
        self.frame_results.pack_forget()
        self.results_button.config(text="Show Export Tools")
        else:
        self.frame_results.pack(fill="both", expand=True, padx=6, pady=6)
        self.results_button.config(text="Hide Export Tools")

    # ------------------- Persistence -------------------
    def autoload_config(self):
        path = self.config_path
        try:
            if os.path.exists(path):
                cfg = load_xml(path)
                for k, v in cfg.items():
                    if k in self.vars:
                        self.vars[k].set(str(v))
                self.log(f"[Loaded config] {path}")
            else:
                self.save_current_to_default()
                self.log(f"[Created default config] {path}")
        except Exception as e:
            self.log(f"[WARN] Failed to load config: {e}")

    def gather_current_config(self) -> Dict[str, Any]:
        return {k: self.vars[k].get() for k, _ in XML_FIELDS}

    def save_current_to_default(self):
        data = self.gather_current_config()
        try:
            parent = os.path.dirname(self.config_path)
            os.makedirs(parent, exist_ok=True)
            save_xml(self.config_path, data)
            self.log(f"[Saved config] {self.config_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save XML:\n{e}")

    # ------------------- UI Callbacks -------------------
    def browse_script(self):
        path = filedialog.askopenfilename(title="Select video_classifier.py",
                                          filetypes=[("Python files", "*.py"), ("All files", "*.*")])
        if path:
            self.script_path.set(path)

    def choose_root(self):
        folder = filedialog.askdirectory(title="Select root folder to scan")
        if folder:
            self.vars["root"].set(folder)
            self.save_current_to_default()

    def choose_output(self):
        path = filedialog.asksaveasfilename(title="Select output CSV",
                                            defaultextension=".csv",
                                            filetypes=[("CSV", "*.csv")])
        if path:
            self.vars["output"].set(path)
            self.save_current_to_default()

    def _toggle_advanced(self):
        if self.advanced_visible.get():
            self.frame_adv.pack(fill="x", padx=8, pady=(0,6))
        else:
            self.frame_adv.forget()

    def on_preflight(self):
        root = self.vars["root"].get().strip()
        out = self.vars["output"].get().strip()
        if not root:
            self.log("[Preflight] Set 'root' first.")
            return
        if not os.path.isdir(root):
            self.log(f"[Preflight] Root not found: {root}")
            return

        self.log(f"[Preflight] Scanning: {root}")
        total = 0
        counts = Counter()
        samples = []

        for r, _, files in os.walk(root):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in VIDEO_EXTS:
                    counts[ext] += 1
                    total += 1
                    if len(samples) < 3:
                        samples.append(os.path.join(r, f))

        if total == 0:
            self.log("[Preflight] Found 0 video files. Check extensions or folder selection.")
        else:
            self.log(f"[Preflight] Found {total} files.")
            for ext, n in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
                self.log(f"  {ext}: {n}")
            for s in samples:
                self.log(f"  sample: {s}")

        if out:
            parent = os.path.dirname(out) or os.getcwd()
            try:
                os.makedirs(parent, exist_ok=True)
                test = os.path.join(parent, "__write_test__")
                with open(test, "w", encoding="utf-8") as fh:
                    fh.write("ok")
                os.remove(test)
                self.log(f"[Preflight] Output folder writable: {parent}")
            except Exception as e:
                self.log(f"[Preflight] Output not writable: {parent} -> {e}")
        else:
            self.log("[Preflight] No output set; set it to confirm write access.")

    def _apply_presets_to_vars(self):
        # Apply sensitivity -> advanced thresholds
        s = self.sensitivity_choice.get()
        if s in SENSITIVITY_PRESETS:
            for k, v in SENSITIVITY_PRESETS[s].items():
                self.vars[k].set(str(v))
        # Apply dedupe mode
        d = self.dedupe_choice.get()
        if d in DEDUP_PRESETS:
            for k, v in DEDUP_PRESETS[d].items():
                self.vars[k].set(str(v))

    def on_run(self):
        if getattr(self, "proc", None) is not None:
            messagebox.showwarning("Running", "Process already running.")
            return

        # Apply beginner presets to the advanced vars before running
        self._apply_presets_to_vars()
        self.save_current_to_default()

        # Gather all settings from the GUI into a dictionary
        config_dict = self.gather_current_config()

        args_obj = Args(config_dict)
        
        # --- Main Task Logic ---
        def task():
            self.proc = True
            self.results_button.state(['disabled']) # Disable button during run
            try:
                # The main() function now needs to return the results
                results_list = run_classifier_logic(args_obj)

                # Store the results
                self.results_data = results_list
                
                self.populate_results_tree()

                # Enable the button
                self.results_button.state(['!disabled'])

                self.log("[Process finished successfully]")
                self.log(f"Found {len(self.results_data)} files. Click 'Show Export Tools' to view and export.")

            except Exception as e:
                self.log(f"[ERROR] {e}")
            finally:
                self.proc = None
                
        # Run the classifier in a separate thread to keep the GUI responsive
        threading.Thread(target=task, daemon=True).start()

    def on_stop(self):
        if getattr(self, "proc", None) is not None:
            try:
                self.proc.terminate()
                self.log("[Sent terminate signal]")
            except Exception as e:
                self.log(f"[ERROR terminating] {e}")
        else:
            self.log("[No process running]")

    def on_close(self):
        try:
            self.save_current_to_default()
        except Exception:
            pass
        if getattr(self, "proc", None) is not None:
            try:
                self.proc.terminate()
            except Exception:
                pass
        self.destroy()

    def log(self, msg: str):
        self.console.insert("end", msg + "\n")
        self.console.see("end")

# needed by Tooltip
from tkinter import Tk  # noqa: E402

if __name__ == "__main__":
    App().mainloop()
