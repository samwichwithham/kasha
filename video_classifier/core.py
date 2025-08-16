#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
    # basic
    path: str
    size_bytes: int
    width: Optional[int]
    height: Optional[int]
    fps: Optional[float]
    frame_count: Optional[int]
    duration_sec: Optional[float]
    codec_note: Optional[str]

    # exif (optional)
    make: Optional[str] = None
    model: Optional[str] = None
    creation_time: Optional[str] = None

    # VAD metrics
    speech_total_sec: Optional[float] = None
    speech_ratio: Optional[float] = None
    longest_speech_sec: Optional[float] = None
    segments_count: Optional[int] = None
    vad_aggressiveness: Optional[int] = None

    # classification
    class_label: Optional[str] = None  # "interview" | "broll" | "other"

    # dedupe
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None
    session_id: Optional[str] = None


# ----------------------------- Helpers -----------------------------

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

def find_videos(root: str) -> List[Path]:
    base = Path(root)
    if not base.exists():
        print(f"[WARN] Root not found: {root}", file=sys.stderr)
        return []
    vids: List[Path] = []
    for p in base.rglob("*"):
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

def classify_interview(v: VideoFeatures, args: argparse.Namespace) -> str:
    # if duration known, basic min filter
    if v.duration_sec is not None and v.duration_sec < float(args.min_duration):
        return "other"
    # VAD-driven rules (use interview thresholds)
    dur = v.duration_sec or 0.0
    st  = v.speech_total_sec or 0.0
    sr  = v.speech_ratio or 0.0
    ls  = v.longest_speech_sec or 0.0
    sc  = v.segments_count or 0

    if (dur >= float(args.min_duration_sec) and
        sr  >= float(args.min_speech_ratio) and
        ls  >= float(args.min_longest_speech_sec) and
        sc  >= int(args.min_segments_count)):
        return "interview"

    # crude b-roll: very low speech or no continuous speech
    if sr <= 0.05 or ls <= 3.0:
        return "broll"

    return "other"

def session_id_for(path: Path, args: argparse.Namespace) -> str:
    if args.session_scope == "global":
        return "global"
    # derive from folder depth (shoot-depth)
    parts = path.parts
    depth = max(1, int(args.shoot_depth))
    # take last `depth` directories (excluding filename)
    dirs = list(parts[:-1])[-depth:]
    return "/".join(dirs) if dirs else "global"

def choose_primary(candidates: List[VideoFeatures]) -> VideoFeatures:
    # Prefer bigger resolution (area), then bigger file size
    def key(v: VideoFeatures):
        area = (v.width or 0) * (v.height or 0)
        return (area, v.size_bytes)
    return sorted(candidates, key=key, reverse=True)[0]

def dedupe_within_sessions(videos: List[VideoFeatures], args: argparse.Namespace) -> None:
    """
    Mark duplicates in-place using session_id + duration proximity.
    """
    tol = float(args.duration_tolerance_sec)
    # group by session
    by_sess: Dict[str, List[VideoFeatures]] = {}
    for v in videos:
        sid = session_id_for(Path(v.path), args)
        v.session_id = sid
        by_sess.setdefault(sid, []).append(v)

    for sid, group in by_sess.items():
        # bucket by rounded duration to nearest second
        buckets: Dict[int, List[VideoFeatures]] = {}
        for v in group:
            d = v.duration_sec if (v.duration_sec is not None) else -1
            buckets.setdefault(int(round(d)), []).append(v)
        # within each bucket, merge near durations
        used = set()
        for dkey, items in buckets.items():
            # try to cluster by duration within tolerance
            items_sorted = sorted(items, key=lambda x: (x.duration_sec or -1))
            clusters: List[List[VideoFeatures]] = []
            for v in items_sorted:
                if any(v is u for cluster in clusters for u in cluster):
                    continue
                cluster = [v]
                for w in items_sorted:
                    if w is v or any(w is u for u in cluster):
                        continue
                    dv = (v.duration_sec or -1) - (w.duration_sec or -1)
                    if abs(dv) <= tol:
                        cluster.append(w)
                clusters.append(cluster)
            # choose a primary per cluster
            for cluster in clusters:
                if len(cluster) <= 1:
                    continue
                primary = choose_primary(cluster)
                for vv in cluster:
                    if vv is not primary:
                        vv.is_duplicate = True
                        vv.duplicate_of = primary.path

# ----------------------------- Per-file processing ---------------------------

def process_one(path: Path, args: argparse.Namespace, do_exif: bool) -> VideoFeatures:
    # Basic CV features
    try:
        size_bytes = path.stat().st_size
    except Exception:
        size_bytes = -1
    w,h,fps,frames,dur,codec_note = analyze_with_cv2(str(path))

    vf = VideoFeatures(
        path=str(path),
        size_bytes=size_bytes,
        width=w, height=h, fps=fps, frame_count=frames, duration_sec=dur,
        codec_note=codec_note
    )

    # EXIF (optional)
    if do_exif:
        meta = run_exiftool(str(path))
        if meta:
            vf.make = meta.get("make")
            vf.model = meta.get("model")
            vf.creation_time = meta.get("creation_time")

    # VAD metrics
    pcm = extract_audio_mono16k_pcm(str(path))
    if pcm:
        st, sr, ls, sc = vad_metrics_from_pcm16k(pcm, aggressiveness=int(args.vad_aggressiveness))
        vf.speech_total_sec = round(st, 2)
        vf.speech_ratio = round(sr, 4)
        vf.longest_speech_sec = round(ls, 2)
        vf.segments_count = int(sc)
        vf.vad_aggressiveness = int(args.vad_aggressiveness)
    else:
        vf.speech_total_sec = 0.0
        vf.speech_ratio = 0.0
        vf.longest_speech_sec = 0.0
        vf.segments_count = 0
        vf.vad_aggressiveness = int(args.vad_aggressiveness)

    # Label
    vf.class_label = classify_interview(vf, args)
    return vf

# --------------------------------- Main --------------------------------------

def main(args: object) -> None:

    print("DEBUG: starting classifier")
    print("DEBUG: root =", args.root)
    print("DEBUG: output =", args.output)

    start = datetime.now()
    vids = find_videos(args.root)
    print(f"DEBUG: candidate files: {len(vids)}")
    for sample in vids[:3]:
        print("DEBUG: sample:", str(sample))

    results: List[VideoFeatures] = []
    do_exif = bool(args.enable_exiftool)

    # Parallel
    workers = max(1, int(args.workers or 1))
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

    # De-duplication (in-place mark)
    if args.dedupe and results:
        dedupe_within_sessions(results, args)

    # Build DataFrame
    rows = [asdict(v) for v in results]
    df = pd.DataFrame(rows)

    # Ensure output dir
    out_dir = os.path.dirname(args.output) or "."
    os.makedirs(out_dir, exist_ok=True)

    # Write ALL files CSV (report.csv at --output)
    df.to_csv(args.output, index=False)
    print("DEBUG: wrote CSV ->", args.output)

    # Write INTERVIEWS (deduped) CSV
    interviews = df[df["class_label"] == "interview"].copy()
    if "is_duplicate" in interviews.columns:
        interviews = interviews[~interviews["is_duplicate"].fillna(False)]
    base, ext = os.path.splitext(args.output)
    interviews_path = f"{base.replace(os.sep, os.sep)[:-0]}interviews.csv" if base else "interviews.csv"
    interviews.to_csv(interviews_path, index=False)
    print("DEBUG: wrote interviews CSV ->", interviews_path, f"(rows={len(interviews)})")

    # Summary text
    try:
        end = datetime.now()
        txt_path = os.path.join(out_dir, "video_classifier_summary.txt")
        with open(txt_path, "w", encoding="utf-8") as fh:
            fh.write(f"Root: {args.root}\n")
            fh.write(f"All files CSV: {args.output} (rows={len(df)})\n")
            fh.write(f"Interviews CSV: {interviews_path} (rows={len(interviews)})\n")
            fh.write(f"Elapsed: {(end-start).total_seconds():.2f}s\n")
            if len(df):
                fh.write("Class counts:\n")
                fh.write(df["class_label"].value_counts(dropna=False).to_string() + "\n")
                if "is_duplicate" in df.columns:
                    fh.write(f"Duplicates marked: {int(df['is_duplicate'].fillna(False).sum())}\n")
        print("DEBUG: wrote summary ->", txt_path)
    except Exception as e:
        print(f"[WARN] summary failed: {e}", file=sys.stderr)

if __name__ == "__main__":
    parsed_args = parse_args()
    main(parsed_args)
