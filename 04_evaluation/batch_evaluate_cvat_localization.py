#!/usr/bin/env python3
"""Batch CVAT evaluation for the multimodal dog fingerprinting pipeline.

This script evaluates all videos that have:
- a top-level CVAT annotations.xml;
- completed pipeline base outputs;
- completed persistent-profile outputs.

It runs evaluate_cvat_barker_identity.py for each video, then aggregates the
valid active-barker localization metrics. Persistent CVAT track IDs are not
treated as physical dog identities in the final aggregate.

Optional manual dog counts
--------------------------
Provide a CSV with columns:

video_id,total_dogs,barking_dogs,silent_dogs

The script will compare these ground-truth counts with pipeline predictions from
pipeline_summary_v2.csv.

Example
-------
python 04_evaluation/batch_evaluate_cvat_localization.py \
  --annotations-root /path/to/annotations \
  --videos-root /path/to/videos \
  --pipeline-output-root /path/to/outputs/cvat_evaluation_runs \
  --output-dir /path/to/outputs/cvat_batch_results \
  --evaluator 04_evaluation/evaluate_cvat_barker_identity.py \
  --spatial-iou-threshold 0.30

With manual counts:

python 04_evaluation/batch_evaluate_cvat_localization.py \
  ... \
  --manual-counts /path/to/manual_dog_counts.csv
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations-root", type=Path, required=True)
    parser.add_argument("--videos-root", type=Path, required=True)
    parser.add_argument("--pipeline-output-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--manual-counts", type=Path)
    parser.add_argument("--spatial-iou-threshold", type=float, default=0.30)
    parser.add_argument("--frame-tolerance", type=int, default=2)
    parser.add_argument(
        "--video-ids",
        nargs="*",
        help="Optional subset of video IDs.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def find_video(videos_root: Path, video_id: str) -> Path | None:
    matches = sorted(videos_root.glob(f"{video_id}_*.mp4"))
    return matches[0] if len(matches) >= 1 else None


def safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, OSError):
        return None


def first_numeric(frame: pd.DataFrame, column: str, default=np.nan):
    if frame is None or frame.empty or column not in frame.columns:
        return default
    value = pd.to_numeric(frame[column], errors="coerce").iloc[0]
    return value if pd.notna(value) else default


def main() -> None:
    args = parse_args()

    annotations_root = args.annotations_root.expanduser().resolve()
    videos_root = args.videos_root.expanduser().resolve()
    pipeline_root = args.pipeline_output_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    evaluator = args.evaluator.expanduser().resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    if not evaluator.is_file():
        raise FileNotFoundError(f"Evaluator not found: {evaluator}")

    annotation_dirs = sorted(
        path
        for path in annotations_root.iterdir()
        if path.is_dir() and (path / "annotations.xml").is_file()
    )

    requested = set(args.video_ids or [])
    if requested:
        annotation_dirs = [
            path for path in annotation_dirs if path.name in requested
        ]

    manual_counts = None
    if args.manual_counts:
        manual_counts = pd.read_csv(args.manual_counts)
        required = {
            "video_id",
            "total_dogs",
            "barking_dogs",
            "silent_dogs",
        }
        missing = required - set(manual_counts.columns)
        if missing:
            raise KeyError(
                f"Manual counts CSV is missing: {sorted(missing)}"
            )
        manual_counts["video_id"] = manual_counts["video_id"].astype(str)

    rows: list[dict] = []
    failures: list[dict] = []

    for annotation_dir in annotation_dirs:
        video_id = annotation_dir.name
        video_path = find_video(videos_root, video_id)

        base_dir = pipeline_root / f"{video_id}_base"
        profile_dir = pipeline_root / f"{video_id}_profiles"
        fusion_dir = pipeline_root / f"{video_id}_fusion"
        video_eval_dir = output_dir / video_id

        predictions_path = base_dir / "predictions.csv"
        tracks_path = base_dir / "tracks.csv"
        profile_map_path = (
            profile_dir / "track_to_all_dog_profile.csv"
        )
        fusion_summary_path = fusion_dir / "pipeline_summary_v2.csv"

        missing_inputs = []
        for label, path in [
            ("video", video_path),
            ("predictions", predictions_path),
            ("tracks", tracks_path),
            ("profile_map", profile_map_path),
        ]:
            if path is None or not Path(path).is_file():
                missing_inputs.append(label)

        if missing_inputs:
            failures.append(
                {
                    "video_id": video_id,
                    "status": "missing_inputs",
                    "details": ",".join(missing_inputs),
                }
            )
            continue

        aggregate_path = video_eval_dir / "aggregate_metrics.csv"

        if args.force or not aggregate_path.is_file():
            video_eval_dir.mkdir(parents=True, exist_ok=True)

            command = [
                sys.executable,
                str(evaluator),
                "--video",
                str(video_path),
                "--annotations-xml",
                str(annotation_dir / "annotations.xml"),
                "--predictions",
                str(predictions_path),
                "--tracks",
                str(tracks_path),
                "--track-profile-map",
                str(profile_map_path),
                "--output-dir",
                str(video_eval_dir),
                "--spatial-iou-threshold",
                str(args.spatial_iou_threshold),
                "--frame-tolerance",
                str(args.frame_tolerance),
            ]

            print("\n" + "=" * 78)
            print("EVALUATING", video_id)
            print("=" * 78)

            try:
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as error:
                failures.append(
                    {
                        "video_id": video_id,
                        "status": "evaluator_failed",
                        "details": str(error),
                    }
                )
                continue

        aggregate = safe_read_csv(aggregate_path)
        if aggregate is None or aggregate.empty:
            failures.append(
                {
                    "video_id": video_id,
                    "status": "missing_evaluation_output",
                    "details": str(aggregate_path),
                }
            )
            continue

        fusion_summary = safe_read_csv(fusion_summary_path)

        row = {
            "video_id": video_id,
            "status": "complete",
            "events_evaluated": int(
                first_numeric(aggregate, "events_evaluated", 0)
            ),
            "localized_correct_events": int(
                first_numeric(
                    aggregate,
                    "localized_correct_events",
                    0,
                )
            ),
            "active_barker_localization_accuracy": first_numeric(
                aggregate,
                "active_barker_localization_accuracy",
            ),
            "mean_spatial_iou": first_numeric(
                aggregate,
                "mean_spatial_iou",
            ),
            "predicted_total_dogs": first_numeric(
                fusion_summary,
                "total_persistent_dogs",
            ),
            "predicted_barking_dogs": first_numeric(
                fusion_summary,
                "barking_dogs",
            ),
            "predicted_silent_dogs": first_numeric(
                fusion_summary,
                "silent_dogs",
            ),
            "predicted_bark_events": first_numeric(
                fusion_summary,
                "bark_events",
            ),
            "assigned_bark_events": first_numeric(
                fusion_summary,
                "assigned_bark_events",
            ),
            "audio_visual_conflicts": first_numeric(
                fusion_summary,
                "audio_visual_conflicts",
            ),
        }

        if manual_counts is not None:
            manual = manual_counts[
                manual_counts["video_id"] == video_id
            ]

            if not manual.empty:
                manual_row = manual.iloc[0]
                row.update(
                    {
                        "gt_total_dogs": int(
                            manual_row["total_dogs"]
                        ),
                        "gt_barking_dogs": int(
                            manual_row["barking_dogs"]
                        ),
                        "gt_silent_dogs": int(
                            manual_row["silent_dogs"]
                        ),
                    }
                )

                if pd.notna(row["predicted_total_dogs"]):
                    row["total_dog_count_error"] = (
                        row["predicted_total_dogs"]
                        - row["gt_total_dogs"]
                    )
                    row["absolute_total_dog_count_error"] = abs(
                        row["total_dog_count_error"]
                    )

                if pd.notna(row["predicted_barking_dogs"]):
                    row["barking_dog_count_error"] = (
                        row["predicted_barking_dogs"]
                        - row["gt_barking_dogs"]
                    )

                if pd.notna(row["predicted_silent_dogs"]):
                    row["silent_dog_count_error"] = (
                        row["predicted_silent_dogs"]
                        - row["gt_silent_dogs"]
                    )

        rows.append(row)

    per_video = pd.DataFrame(rows)
    failures_frame = pd.DataFrame(failures)

    if per_video.empty:
        raise SystemExit(
            "No videos were successfully evaluated. "
            "Check missing inputs in batch_failures.csv."
        )

    per_video = per_video.sort_values(
        "video_id",
        key=lambda values: pd.to_numeric(
            values,
            errors="coerce",
        ),
    ).reset_index(drop=True)

    total_events = int(per_video["events_evaluated"].sum())
    total_correct = int(
        per_video["localized_correct_events"].sum()
    )

    aggregate_row = {
        "videos_evaluated": len(per_video),
        "videos_failed_or_skipped": len(failures_frame),
        "events_evaluated": total_events,
        "localized_correct_events": total_correct,
        "micro_localization_accuracy": (
            total_correct / total_events
            if total_events
            else np.nan
        ),
        "macro_localization_accuracy": float(
            per_video[
                "active_barker_localization_accuracy"
            ].mean()
        ),
        "mean_spatial_iou": float(
            np.average(
                per_video["mean_spatial_iou"],
                weights=per_video["events_evaluated"],
            )
        )
        if total_events
        else np.nan,
        "predicted_total_dogs": float(
            per_video["predicted_total_dogs"].sum()
        ),
        "predicted_barking_dogs": float(
            per_video["predicted_barking_dogs"].sum()
        ),
        "predicted_silent_dogs": float(
            per_video["predicted_silent_dogs"].sum()
        ),
        "predicted_bark_events": float(
            per_video["predicted_bark_events"].sum()
        ),
        "assigned_bark_events": float(
            per_video["assigned_bark_events"].sum()
        ),
    }

    if manual_counts is not None and "gt_total_dogs" in per_video.columns:
        valid_counts = per_video.dropna(
            subset=["gt_total_dogs", "predicted_total_dogs"]
        )

        aggregate_row.update(
            {
                "videos_with_manual_counts": len(valid_counts),
                "gt_total_dogs": float(
                    valid_counts["gt_total_dogs"].sum()
                ),
                "gt_barking_dogs": float(
                    valid_counts["gt_barking_dogs"].sum()
                ),
                "gt_silent_dogs": float(
                    valid_counts["gt_silent_dogs"].sum()
                ),
                "total_dog_count_bias": float(
                    valid_counts["total_dog_count_error"].mean()
                ),
                "total_dog_count_mae": float(
                    valid_counts[
                        "absolute_total_dog_count_error"
                    ].mean()
                ),
                "exact_total_dog_count_rate": float(
                    (
                        valid_counts["total_dog_count_error"]
                        == 0
                    ).mean()
                ),
                "overall_dog_overestimation_fraction": (
                    (
                        valid_counts["predicted_total_dogs"].sum()
                        - valid_counts["gt_total_dogs"].sum()
                    )
                    / valid_counts["gt_total_dogs"].sum()
                )
                if valid_counts["gt_total_dogs"].sum()
                else np.nan,
            }
        )

    aggregate = pd.DataFrame([aggregate_row])

    per_video.to_csv(
        output_dir / "per_video_cvat_results.csv",
        index=False,
    )
    aggregate.to_csv(
        output_dir / "aggregate_cvat_results.csv",
        index=False,
    )
    failures_frame.to_csv(
        output_dir / "batch_failures.csv",
        index=False,
    )

    result = aggregate.iloc[0]
    summary_lines = [
        "BATCH CVAT ACTIVE-BARKER EVALUATION",
        "",
        f"Videos evaluated: {int(result['videos_evaluated'])}",
        f"Videos failed/skipped: {int(result['videos_failed_or_skipped'])}",
        f"Bark events evaluated: {int(result['events_evaluated'])}",
        (
            "Micro localization accuracy: "
            f"{100 * result['micro_localization_accuracy']:.2f}%"
        ),
        (
            "Macro localization accuracy: "
            f"{100 * result['macro_localization_accuracy']:.2f}%"
        ),
        f"Mean spatial IoU: {result['mean_spatial_iou']:.4f}",
    ]

    if "total_dog_count_mae" in result.index:
        summary_lines.extend(
            [
                "",
                "Dog-count evaluation",
                (
                    "Videos with manual counts: "
                    f"{int(result['videos_with_manual_counts'])}"
                ),
                (
                    "Total dog count MAE: "
                    f"{result['total_dog_count_mae']:.4f}"
                ),
                (
                    "Mean signed count error: "
                    f"{result['total_dog_count_bias']:.4f}"
                ),
                (
                    "Exact count rate: "
                    f"{100 * result['exact_total_dog_count_rate']:.2f}%"
                ),
                (
                    "Overall overestimation fraction: "
                    f"{100 * result['overall_dog_overestimation_fraction']:.2f}%"
                ),
            ]
        )

    summary = "\n".join(summary_lines) + "\n"
    (output_dir / "batch_summary.txt").write_text(
        summary,
        encoding="utf-8",
    )

    print("\n" + summary)
    print("Saved:", output_dir)


if __name__ == "__main__":
    main()
