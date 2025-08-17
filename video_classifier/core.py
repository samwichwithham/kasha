#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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


"""
Video interview detector with de-duplication.

Outputs:
- report.csv (all files & metrics)
- interviews.csv (only interviews, duplicates removed)
- video_classifier_summary.txt

Dependencies:
  brew install ffmpeg exiftool   # exiftool optional
  pip install webrtcvad numpy pandas opencv-python
"""

from __future__ import annotations
import os, sys, argparse, xml.etree.ElementTree as ET, subprocess, math, json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import cv2

# ----------------------------- Config / Constants -----------------------------

VIDEO_EXTS = {
    ".mp4", ".mov", ".m4v", ".avi", ".mts", ".mxf", ".mkv", ".wmv", ".mpg", ".mpeg", ".3gp", ".m2ts"
}

# Default interview thresholds (overridable by XML/flags)
DEFAULTS = dict(
    min_duration_sec=120.0,
    min_speech_ratio=0.25,
    min_longest_speech_sec=20.0,
    min_segments_count=3,
    vad_aggressiveness=2,           # 0-3 (higher = stricter speech)
    duration_tolerance_sec=3.0,     # for duplicate grouping
)

@dataclass
class VideoFeatures:
    # Core Info
    path: str
    project: Optional[str] = None

    # Video Properties
    resolution: Optional[str] = None
    fps: Optional[float] = None
    duration_min: Optional[float] = None
    size_gb: Optional[float] = None

    # Classification
    class_label: Optional[str] = None  # interview | broll | other
    is_log: bool = False

    # Metadata
    camera_model: Optional[str] = None
    creation_time: Optional[str] = None

    # VAD Metrics (kept for reference, won't be in final CSV unless needed)
    speech_ratio: Optional[float] = None
    longest_speech_sec: Optional[float] = None

    # Duplicates (to be removed, placeholder for now)
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None
    session_id: Optional[str] = None


# ----------------------------- Helpers -----------------------------
def parse_camera_from_filename(filename: str) -> Optional[str]:
    """
    Parses a filename to find common camera model identifiers.
    Returns the first match found or None.
    """
    # A list of common camera models to search for (case-insensitive)
    # You can expand this list with other models you use.
    known_cameras = ["fx30", "fx3", "fx6", "a7siii", "a7s3"]

    lower_filename = filename.lower()

    for camera in known_cameras:
        if camera in lower_filename:
            # Standardize the name (e.g., a7siii -> a7s3)
            if camera == "a7siii":
                return "a7s3"
            return camera.upper() # Return in uppercase for consistency, e.g., FX30

    return None

def is_video(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTS

def parse_xml_config(xml_path: str) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    if not xml_path:
        return cfg
    try:
        if not os.path.exists(xml_path):
            print(f"[WARN] XML config not found: {xml_path}", file=sys.stderr)
            return cfg
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for child in root:
            key = child.tag.strip()
            value = (child.text or "").strip()
            if key:
                cfg[key] = value
        print(f"[DEBUG] Loaded XML defaults from {xml_path}")
    except Exception as e:
        print(f"[WARN] Failed to parse XML config: {e}", file=sys.stderr)
    return cfg

def coerce_cfg_types(cfg: Dict[str, Any]) -> Dict[str, Any]:
    typed = dict(cfg)
    float_keys = {"interval","vad_window","vad_thresh","vad_min_speech","vad_max_gap",
                  "short_max","long_min","min_duration","session_overlap","log_img_thresh",
                  "min_duration_sec","min_speech_ratio","min_longest_speech_sec",
                  "duration_tolerance_sec"}
    int_keys = {"max_frames","workers","shoot_depth","session_gap_tol","min_segments_count","vad_aggressiveness"}
    bool_keys = {"dedupe","enable_exiftool"}

    for k in list(typed.keys()):
        v = typed[k]
        if k in float_keys:
            try: typed[k] = float(v)
            except: pass
        elif k in int_keys:
            try: typed[k] = int(float(v))
            except: pass
        elif k in bool_keys:
            typed[k] = str(v).lower() in {"1","true","yes","on"}
    return typed

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interview detector with de-duplication")
    # keep existing GUI flags
    p.add_argument("--config", type=str, default="", help="XML config path (optional)")
    p.add_argument("--root", type=str, required=False)
    p.add_argument("--output", type=str, required=False)

    p.add_argument("--interval", type=float, default=2.0)
    p.add_argument("--max-frames", type=int, default=400)
    p.add_argument("--vad-window", type=float, default=0.5)
    p.add_argument("--vad-thresh", type=float, default=0.5)
    p.add_argument("--vad-min-speech", type=float, default=0.3)
    p.add_argument("--vad-max-gap", type=float, default=0.5)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--short-max", type=float, default=5.0)
    p.add_argument("--long-min", type=float, default=8.0)
    p.add_argument("--min-duration", type=float, default=0.5)
    p.add_argument("--session-scope", type=str, default="shoot", choices=["shoot", "global"])
    p.add_argument("--shoot-depth", type=int, default=2)
    p.add_argument("--session-overlap", type=float, default=0.30)
    p.add_argument("--session-gap-tol", type=float, default=15.0)
    p.add_argument("--log-img-thresh", type=float, default=0.55)
    p.add_argument("--enable-exiftool", action="store_true")
    p.add_argument("--dedupe", dest="dedupe", action="store_true", help="Enable de-duplication (default)")
    p.add_argument("--no-dedupe", dest="dedupe", action="store_false", help="Disable de-duplication")
    p.set_defaults(dedupe=True)

    # Interview classification thresholds
    p.add_argument("--min-duration-sec", type=float, default=DEFAULTS["min_duration_sec"])
    p.add_argument("--min-speech-ratio", type=float, default=DEFAULTS["min_speech_ratio"])
    p.add_argument("--min-longest-speech-sec", type=float, default=DEFAULTS["min_longest_speech_sec"])
    p.add_argument("--min-segments-count", type=int, default=DEFAULTS["min_segments_count"])
    p.add_argument("--vad-aggressiveness", type=int, default=DEFAULTS["vad_aggressiveness"])
    p.add_argument("--duration-tolerance-sec", type=float, default=DEFAULTS["duration_tolerance_sec"])

    # Apply config defaults first (so CLI overrides)
    pre = p.parse_known_args()[0]
    xml_cfg = coerce_cfg_types(parse_xml_config(pre.config)) if pre.config else {}
    if xml_cfg:
        p.set_defaults(**xml_cfg)

    args = p.parse_args()

    if not args.root or not args.output:
        p.error("Missing --root or --output (can come from XML or CLI).")
    return args

# ----------------------------- Discovery / EXIF ------------------------------

def find_videos(root: str) -> list[Path]:
    """
    Recursively finds all video files in a directory, skipping specified folders.
    """
    base = Path(root)
    if not base.exists():
        print(f"[WARN] Root not found: {root}", file=sys.stderr)
        return []

    vids: list[Path] = []
    folders_to_skip = {"proxy", "proxies"} # Case-insensitive check

    for p in base.rglob("*"):
        # To improve efficiency, check if any part of the path is a skip folder
        if any(part.lower() in folders_to_skip for part in p.parts):
            continue

        if is_video(p):
            vids.append(p)

    return vids

def run_exiftool(path: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        proc = subprocess.Popen(["exiftool", "-json", "-n", path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, _ = proc.communicate(timeout=30)
        if proc.returncode != 0:
            return out
        data = json.loads(stdout)
        if isinstance(data, list) and data:
            d = data[0]
            out["make"] = d.get("Make")
            out["model"] = d.get("Model")
            out["creation_time"] = d.get("CreateDate") or d.get("MediaCreateDate") or d.get("DateTimeOriginal")
    except Exception:
        pass
    return out

# ----------------------------- OpenCV basics ---------------------------------

def analyze_with_cv2(path: str) -> Tuple[Optional[int], Optional[int], Optional[float], Optional[int], Optional[float], Optional[str]]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None, None, None, None, None, "open_failed"
    try:
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
        fps    = float(cap.get(cv2.CAP_PROP_FPS)) or None
        count  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        dur    = (count / fps) if (fps and fps > 0 and count) else None
        return width, height, fps, count, dur, None
    finally:
        cap.release()

# ----------------------------- Audio / VAD -----------------------------------

def extract_audio_mono16k_pcm(path: str) -> Optional[bytes]:
    """
    Returns raw 16k mono PCM (16-bit) bytes, or None.
    Uses ffmpeg to decode audio quickly.
    """
    cmd = [
        "ffmpeg", "-v", "error",
        "-i", path,
        "-f", "s16le", "-acodec", "pcm_s16le",
        "-ac", "1", "-ar", "16000",
        "-"
    ]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        pcm, _ = proc.communicate(timeout=120)
        if proc.returncode != 0 or not pcm:
            return None
        return pcm
    except Exception:
        return None

def vad_metrics_from_pcm16k(pcm: bytes, aggressiveness: int = 2, frame_ms: int = 30) -> Tuple[float, float, float, int]:
    """
    WebRTC VAD over mono 16k PCM s16le.
    Returns: (speech_total_sec, speech_ratio, longest_speech_sec, segments_count)
    """
    import webrtcvad
    vad = webrtcvad.Vad(int(max(0, min(3, aggressiveness))))

    sample_rate = 16000
    frame_len = int(sample_rate * (frame_ms / 1000.0))  # samples/frame
    bytes_per_sample = 2
    step = frame_len * bytes_per_sample

    n_frames = len(pcm) // step
    speech_flags: List[bool] = []
    for i in range(n_frames):
        frame = pcm[i*step:(i+1)*step]
        if len(frame) < step:
            break
        is_speech = vad.is_speech(frame, sample_rate)
        speech_flags.append(is_speech)

    # Aggregate contiguous speech segments
    total_speech = 0.0
    longest = 0.0
    segs = 0
    cur = 0
    for flag in speech_flags:
        if flag:
            cur += 1
        else:
            if cur > 0:
                segs += 1
                dur = cur * (frame_ms / 1000.0)
                total_speech += dur
                longest = max(longest, dur)
                cur = 0
    if cur > 0:
        segs += 1
        dur = cur * (frame_ms / 1000.0)
        total_speech += dur
        longest = max(longest, dur)

    duration_sec = len(speech_flags) * (frame_ms / 1000.0)
    ratio = (total_speech / duration_sec) if duration_sec > 0 else 0.0
    return total_speech, ratio, longest, segs

# ----------------------------- Classification / Dedupe -----------------------

def classify_interview(v: VideoFeatures, args: object) -> str:
    # Get the thresholds from the args object, falling back to DEFAULTS
    min_dur_sec = float(getattr(args, 'min_duration_sec', DEFAULTS['min_duration_sec']))
    min_speech_ratio = float(getattr(args, 'min_speech_ratio', DEFAULTS['min_speech_ratio']))
    min_longest_speech = float(getattr(args, 'min_longest_speech_sec', DEFAULTS['min_longest_speech_sec']))
    min_segments = int(getattr(args, 'min_segments_count', DEFAULTS['min_segments_count']))

    # Get the actual values from the video
    dur_sec = (v.duration_min or 0.0) * 60
    sr = v.speech_ratio or 0.0
    ls = v.longest_speech_sec or 0.0
    # The 'segments_count' was missing from our new VideoFeatures, so we'll get it from the VAD analysis directly if available
    # We need to add it back to VideoFeatures and process_one for a permanent fix, but this works for now.
    # This part requires a bit more refactoring later.

    # --- DEBUG PRINT ---
    print(f"\n[DEBUG CLASSIFY] File: {os.path.basename(v.path)}")
    print(f"  - Duration: {dur_sec:.1f}s (Threshold: >{min_dur_sec}s) -> {'OK' if dur_sec >= min_dur_sec else 'FAIL'}")
    print(f"  - Speech Ratio: {sr:.2f} (Threshold: >{min_speech_ratio}) -> {'OK' if sr >= min_speech_ratio else 'FAIL'}")
    print(f"  - Longest Speech: {ls:.1f}s (Threshold: >{min_longest_speech}) -> {'OK' if ls >= min_longest_speech else 'FAIL'}")
    # print(f"  - Segments: {sc} (Threshold: >{min_segments}) -> {'OK' if sc >= min_segments else 'FAIL'}")


    if (dur_sec >= min_dur_sec and
        sr >= min_speech_ratio and
        ls >= min_longest_speech):
        # sc >= min_segments): # Temporarily disabling segment count check
        return "interview"

    # crude b-roll: very low speech or no continuous speech
    if sr <= 0.05 or ls <= 3.0:
        return "broll"

    return "other"

def get_project_name(path: Path, args: object) -> str:
    """
    Derives a project name from the parent folder structure.
    Uses `shoot_depth` to determine how many parent folders to include.
    """
    if getattr(args, 'session_scope', 'shoot') == "global":
        return "global"

    # Get the number of parent directories to use for the project name
    depth = int(getattr(args, 'shoot_depth', 2))

    # Get all parent directory names, excluding the file itself
    parent_dirs = list(path.parent.parts)

    # Take the last `depth` directories
    # Example: /.../23JAN_shoot/CAM1 -> ["23JAN_shoot", "CAM1"]
    # If depth is 1, it will take "CAM1". If 2, it will take "23JAN_shoot/CAM1"
    # Based on your structure, you may want to adjust the depth in the GUI.
    project_parts = parent_dirs[-depth:]

    return "/".join(project_parts) if project_parts else "global"

# ----------------------------- Per-file processing ---------------------------

def process_one(path: Path, args: argparse.Namespace, do_exif: bool) -> VideoFeatures:
    # Basic CV features
    try:
        size_bytes = path.stat().st_size
    except Exception:
        size_bytes = -1
    w,h,fps,frames,dur,codec_note = analyze_with_cv2(str(path))

    # --- Inside process_one function, after w,h,fps,frames,dur are calculated ---

    # Format new fields
    resolution_str = f"{w}x{h}" if w and h else None
    duration_min_val = round(dur / 60, 2) if dur is not None else None
    size_gb_val = round(size_bytes / (1024**3), 2) if size_bytes > 0 else 0.0
    fps_val = round(fps, 2) if fps is not None else None

    vf = VideoFeatures(
        path=str(path),
        size_gb=size_gb_val,
        resolution=resolution_str,
        fps=fps_val,
        duration_min=duration_min_val,
        # codec_note is removed for now
    )

    # EXIF (optional) - update to use new camera_model field
    # --- Inside process_one, replace the old EXIF block with this ---

    # Metadata
    if do_exif:
        meta = run_exiftool(str(path))
        if meta:
            vf.camera_model = meta.get("model")
            vf.creation_time = meta.get("creation_time")

    # If camera model is still unknown, try parsing the filename as a fallback
    if not vf.camera_model:
        vf.camera_model = parse_camera_from_filename(path.name)

    # VAD metrics - update to use new fields
    pcm = extract_audio_mono16k_pcm(str(path))
    if pcm:
        st, sr, ls, sc = vad_metrics_from_pcm16k(pcm, aggressiveness=int(args.vad_aggressiveness))
        vf.speech_ratio = round(sr, 4)
        vf.longest_speech_sec = round(ls, 2)
    else:
        vf.speech_ratio = 0.0
        vf.longest_speech_sec = 0.0

    # Label
    vf.class_label = classify_interview(vf, args)
    
    # Add Project Name
    vf.project = get_project_name(path, args)
    
    return vf

# --------------------------------- Main --------------------------------------
def main(args: object) -> None:
    print("DEBUG: starting classifier")
    print("DEBUG: root =", getattr(args, 'root', 'Not Set'))
    print("DEBUG: output =", getattr(args, 'output', 'Not Set'))

    start = datetime.now()
    vids = find_videos(getattr(args, 'root', ''))
    print(f"DEBUG: candidate files: {len(vids)}")
    for sample in vids[:3]:
        print("DEBUG: sample:", str(sample))

    results: list[VideoFeatures] = []
    do_exif = bool(getattr(args, 'enable_exiftool', False))

    # Parallel
    workers = max(1, int(getattr(args, 'workers', 1)))
    if vids:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(process_one, p, args, do_exif): p for p in vids}
            for fut in as_completed(futs):
                try:
                    v = fut.result()
                    results.append(v)
                    # lightweight live log
                    print(f"DEBUG: done {os.path.basename(v.path)} label={v.class_label} dup={v.is_duplicate}")
                except Exception as e:
                    p = futs.get(fut)
                    print(f"[WARN] processing failed: {p} -> {e}", file=sys.stderr)

    # Build DataFrame
    rows = [asdict(v) for v in results]
    if not rows:
        print("DEBUG: No video files processed. Skipping CSV creation.")
        return

    df = pd.DataFrame(rows)

    # Define final columns for the CSV report
    final_columns = [
        "path", "project", "class_label", "resolution", "duration_min",
        "fps", "size_gb", "camera_model", "creation_time", "is_log"
    ]

    # Filter the DataFrame to only include the columns we want
    df_final = df[[col for col in final_columns if col in df.columns]]

    # Rename columns for readability
    df_final = df_final.rename(columns={
        "duration_min": "Duration (min)",
        "size_gb": "Size (GB)",
        "class_label": "Classification",
        "camera_model": "Camera",
        "creation_time": "Timestamp",
        "is_log": "LOG"
    })

    # Ensure output dir
    output_path = getattr(args, 'output', '')
    out_dir = os.path.dirname(output_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    # Write ALL files CSV (report.csv at --output)
    df_final.to_csv(output_path, index=False)
    print("DEBUG: wrote CSV ->", output_path)

    # --- Inside main, replace the "Write INTERVIEWS (deduped) CSV" block with this ---

    # Write INTERVIEWS CSV for WhisperX
    # Filter the DataFrame to get only rows classified as 'interview'
    interviews_df = df[df["class_label"] == "interview"].copy()

    # Select only the 'path' column for the final CSV
    interview_paths_df = interviews_df[["path"]]

    # Save the list of paths to a new CSV file
    if output_path:
        base, ext = os.path.splitext(output_path)
        # Name the new file `..._interviews.csv`
        interviews_csv_path = f"{base}_interviews.csv"

        interview_paths_df.to_csv(interviews_csv_path, index=False, header=True)
        print("DEBUG: wrote interviews CSV for WhisperX ->", interviews_csv_path, f"(rows={len(interview_paths_df)})")

    # Update the summary text logic
    try:
        end = datetime.now()
        txt_path = os.path.join(out_dir, "video_classifier_summary.txt")
        with open(txt_path, "w", encoding="utf-8") as fh:
            fh.write(f"Root: {getattr(args, 'root', '')}\n")
            fh.write(f"All files CSV: {output_path} (rows={len(df_final)})\n")
            if 'interviews_csv_path' in locals():
                fh.write(f"Interviews CSV: {interviews_csv_path} (rows={len(interview_paths_df)})\n")
            fh.write(f"Elapsed: {(end-start).total_seconds():.2f}s\n")
            if len(df):
                fh.write("Class counts:\n")
                fh.write(df["class_label"].value_counts(dropna=False).to_string() + "\n")
        print("DEBUG: wrote summary ->", txt_path)
    except Exception as e:
        print(f"[WARN] summary failed: {e}", file=sys.stderr)

if __name__ == "__main__":
    parsed_args = parse_args()
    main(parsed_args)
