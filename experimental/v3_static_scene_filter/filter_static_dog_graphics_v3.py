#!/usr/bin/env python3
"""Detect and remove likely static intro/outro dog graphics before profiling.

Experimental v3 preprocessing stage.

Outputs:
- tracks_live_scene_v3.csv
- boundary_scene_samples.csv
- static_scene_intervals.csv
- static_scene_track_decisions.csv
- static_scene_summary.json
- review_frames/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--tracks", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--boundary-window-sec", type=float, default=20.0)
    p.add_argument("--analysis-step-sec", type=float, default=0.5)
    p.add_argument("--static-motion-threshold", type=float, default=3.0)
    p.add_argument("--min-static-duration-sec", type=float, default=3.0)
    p.add_argument("--min-simultaneous-dog-tracks", type=int, default=2)
    p.add_argument("--track-static-fraction-threshold", type=float, default=0.80)
    p.add_argument("--max-live-motion-for-removal", type=float, default=4.0)
    p.add_argument("--cut-difference-threshold", type=float, default=18.0)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def safe_gray(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (320, 180), interpolation=cv2.INTER_AREA)


def frame_difference(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(cv2.absdiff(a, b)))


def sample_boundary_frames(
    video_path: Path,
    boundary_window_sec: float,
    analysis_step_sec: float,
) -> tuple[pd.DataFrame, float, int, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0.0

    intro_end = min(boundary_window_sec, duration_sec)
    outro_start = max(0.0, duration_sec - boundary_window_sec)

    sample_times = sorted(
        {
            round(float(x), 6)
            for x in (
                list(np.arange(0.0, intro_end + 1e-9, analysis_step_sec))
                + list(np.arange(outro_start, duration_sec + 1e-9, analysis_step_sec))
            )
            if 0.0 <= x <= duration_sec
        }
    )

    records: list[dict[str, Any]] = []
    previous_gray = None
    previous_time = None

    for time_sec in sample_times:
        frame_index = min(total_frames - 1, max(0, int(round(time_sec * fps))))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            continue

        gray = safe_gray(frame)
        diff = np.nan
        if previous_gray is not None and previous_time is not None:
            if time_sec - previous_time <= analysis_step_sec * 1.5:
                diff = frame_difference(previous_gray, gray)

        records.append(
            {
                "time_sec": time_sec,
                "frame": frame_index,
                "frame_difference": diff,
                "boundary_side": "intro" if time_sec <= boundary_window_sec else "outro",
            }
        )
        previous_gray = gray
        previous_time = time_sec

    cap.release()
    return pd.DataFrame(records), fps, total_frames, duration_sec


def add_active_track_counts(
    tracks: pd.DataFrame,
    samples: pd.DataFrame,
    fps: float,
    step_sec: float,
) -> pd.DataFrame:
    tolerance = max(1, int(round(step_sec * fps / 2.0)))
    result = samples.copy()
    counts = []

    for row in result.itertuples(index=False):
        nearby = tracks[
            tracks["frame"].between(
                int(row.frame) - tolerance,
                int(row.frame) + tolerance,
            )
        ]
        counts.append(int(nearby["track_id"].nunique()))

    result["active_track_count"] = counts
    return result


def detect_intervals(samples: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for side, group in samples.groupby("boundary_side", sort=False):
        group = group.sort_values("time_sec").reset_index(drop=True)
        diffs = pd.to_numeric(group["frame_difference"], errors="coerce")

        low_motion = diffs <= args.static_motion_threshold
        multi_track = group["active_track_count"] >= args.min_simultaneous_dog_tracks

        cut_support = pd.Series(False, index=group.index)
        for cut_index in group.index[diffs >= args.cut_difference_threshold].tolist():
            cut_support.iloc[cut_index:min(len(group), cut_index + 4)] = True

        accepted = low_motion & (multi_track | cut_support)

        start_idx = None
        for i, is_static in enumerate(accepted.tolist()):
            if is_static and start_idx is None:
                start_idx = i

            is_last = i == len(group) - 1
            if start_idx is not None and ((not is_static) or is_last):
                end_idx = i if (is_static and is_last) else i - 1
                interval = group.iloc[start_idx:end_idx + 1]

                start_time = float(interval["time_sec"].min())
                end_time = float(interval["time_sec"].max())
                duration = end_time - start_time + args.analysis_step_sec

                if duration >= args.min_static_duration_sec:
                    rows.append(
                        {
                            "boundary_side": side,
                            "start_time_sec": start_time,
                            "end_time_sec": end_time,
                            "duration_sec": duration,
                            "start_frame": int(interval["frame"].min()),
                            "end_frame": int(interval["frame"].max()),
                            "median_frame_difference": float(
                                pd.to_numeric(
                                    interval["frame_difference"],
                                    errors="coerce",
                                ).median()
                            ),
                            "max_active_track_count": int(
                                interval["active_track_count"].max()
                            ),
                            "median_active_track_count": float(
                                interval["active_track_count"].median()
                            ),
                        }
                    )
                start_idx = None

    return pd.DataFrame(rows)


def classify_tracks(
    tracks: pd.DataFrame,
    intervals: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    decisions: list[dict[str, Any]] = []

    for track_id, group in tracks.groupby("track_id", sort=True):
        frames = group["frame"].astype(int).to_numpy()
        inside_mask = np.zeros(len(group), dtype=bool)
        sides: list[str] = []

        for interval in intervals.itertuples(index=False):
            current = (
                (frames >= int(interval.start_frame))
                & (frames <= int(interval.end_frame))
            )
            if current.any():
                inside_mask |= current
                sides.append(str(interval.boundary_side))

        static_fraction = float(inside_mask.mean()) if len(group) else 0.0
        matching = intervals[
            intervals.apply(
                lambda row: not (
                    float(group["time_sec"].max()) < float(row["start_time_sec"])
                    or float(group["time_sec"].min()) > float(row["end_time_sec"])
                ),
                axis=1,
            )
        ]

        median_scene_motion = (
            float(matching["median_frame_difference"].median())
            if not matching.empty
            else np.nan
        )

        remove = (
            static_fraction >= args.track_static_fraction_threshold
            and np.isfinite(median_scene_motion)
            and median_scene_motion <= args.max_live_motion_for_removal
        )

        decisions.append(
            {
                "track_id": int(track_id),
                "rows": len(group),
                "first_time_sec": float(group["time_sec"].min()),
                "last_time_sec": float(group["time_sec"].max()),
                "static_interval_fraction": static_fraction,
                "matching_boundary_sides": ",".join(sorted(set(sides))),
                "median_static_scene_motion": median_scene_motion,
                "decision": "remove_static_boundary_graphic" if remove else "keep",
                "remove_track": remove,
            }
        )

    return pd.DataFrame(decisions)


def save_review_frames(
    video_path: Path,
    tracks: pd.DataFrame,
    intervals: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS))

    for interval_index, interval in intervals.iterrows():
        sample_times = [
            float(interval["start_time_sec"]),
            float((interval["start_time_sec"] + interval["end_time_sec"]) / 2.0),
            float(interval["end_time_sec"]),
        ]

        for sample_index, time_sec in enumerate(sample_times):
            frame_index = int(round(time_sec * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                continue

            rows = tracks[
                tracks["frame"].between(frame_index - 1, frame_index + 1)
            ]

            for _, row in rows.iterrows():
                x1, y1, x2, y2 = map(
                    int,
                    [row["x1"], row["y1"], row["x2"], row["y2"]],
                )
                track_id = int(row["track_id"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(
                    frame,
                    f"track {track_id}",
                    (x1, max(24, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.putText(
                frame,
                f"STATIC {interval['boundary_side']} CANDIDATE | {time_sec:.2f}s",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            destination = output_dir / (
                f"interval_{interval_index:02d}_"
                f"{interval['boundary_side']}_sample_{sample_index}.jpg"
            )
            cv2.imwrite(str(destination), frame)

    cap.release()


def main() -> None:
    args = parse_args()
    video_path = args.video.expanduser().resolve()
    tracks_path = args.tracks.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    filtered_path = output_dir / "tracks_live_scene_v3.csv"
    if filtered_path.exists() and not args.force:
        raise FileExistsError(
            f"{filtered_path} exists. Use --force to overwrite."
        )

    tracks = pd.read_csv(tracks_path)

    samples, fps, total_frames, duration_sec = sample_boundary_frames(
        video_path,
        args.boundary_window_sec,
        args.analysis_step_sec,
    )
    samples = add_active_track_counts(
        tracks,
        samples,
        fps,
        args.analysis_step_sec,
    )
    intervals = detect_intervals(samples, args)
    decisions = classify_tracks(tracks, intervals, args)

    remove_ids = set(
        decisions.loc[
            decisions["remove_track"] == True,  # noqa: E712
            "track_id",
        ].astype(int)
    )

    filtered = tracks[
        ~tracks["track_id"].astype(int).isin(remove_ids)
    ].copy()

    filtered.to_csv(filtered_path, index=False)
    samples.to_csv(output_dir / "boundary_scene_samples.csv", index=False)
    intervals.to_csv(output_dir / "static_scene_intervals.csv", index=False)
    decisions.to_csv(
        output_dir / "static_scene_track_decisions.csv",
        index=False,
    )

    save_review_frames(
        video_path,
        tracks,
        intervals,
        output_dir / "review_frames",
    )

    summary = {
        "version": "v3_static_scene_filter",
        "video": str(video_path),
        "duration_sec": duration_sec,
        "fps": fps,
        "original_tracks": int(tracks["track_id"].nunique()),
        "removed_tracks": len(remove_ids),
        "remaining_tracks": int(filtered["track_id"].nunique()),
        "removed_track_ids": sorted(remove_ids),
        "detected_static_intervals": len(intervals),
    }
    (output_dir / "static_scene_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nStatic-scene filtering summary")
    print("=" * 72)
    print("Original temporary tracks:", summary["original_tracks"])
    print("Detected static intervals:", summary["detected_static_intervals"])
    print("Removed tracks:", summary["removed_tracks"])
    print("Removed track IDs:", summary["removed_track_ids"])
    print("Remaining tracks:", summary["remaining_tracks"])
    print("\nFiltered tracks:", filtered_path)


if __name__ == "__main__":
    main()
