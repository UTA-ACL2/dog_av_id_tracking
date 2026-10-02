#!/usr/bin/env python3
"""Evaluate active-barker localization and persistent-profile consistency using CVAT XML.

This script compares:
- full-video CVAT annotations.xml
- pipeline predictions.csv or bark_to_profile_v2.csv
- pipeline tracks.csv
- track_to_all_dog_profile.csv

Core evaluation
---------------
For each predicted bark event:
1. Convert its timestamp interval to video frames.
2. Retrieve the pipeline bounding box for the predicted temporary track.
3. Retrieve all manually annotated barking-dog boxes in that interval.
4. Match the prediction to the ground-truth box with the highest spatial IoU.
5. Count the event as correctly localized when IoU >= --spatial-iou-threshold.

The script also aligns arbitrary persistent-profile labels to CVAT track IDs with
the Hungarian algorithm and reports profile consistency metrics.

Important limitation
--------------------
CVAT track IDs are treated as ground-truth identities only if the same physical
dog keeps the same track ID throughout the full-video annotation. If a new CVAT track was created for each bark event, the localization metrics
remain valid, but persistent-identity accuracy, profile purity, and fragmentation
are not true identity metrics. Inspect cvat_track_summary.csv before reporting those
identity-level results.

Example
-------
python evaluate_cvat_barker_identity.py \
  --video /path/to/video.mp4 \
  --annotations-xml /path/to/video/annotations.xml \
  --predictions /path/to/predictions.csv \
  --tracks /path/to/tracks.csv \
  --track-profile-map /path/to/track_to_all_dog_profile.csv \
  --output-dir /path/to/evaluation
"""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--annotations-xml", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--track-profile-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--spatial-iou-threshold", type=float, default=0.30)
    parser.add_argument(
        "--frame-tolerance",
        type=int,
        default=2,
        help="Search this many frames around each event frame.",
    )
    parser.add_argument(
        "--prediction-track-col",
        default="predicted_track_id",
    )
    parser.add_argument(
        "--prediction-start-col",
        default="start_time_sec",
    )
    parser.add_argument(
        "--prediction-end-col",
        default="end_time_sec",
    )
    parser.add_argument(
        "--prediction-event-col",
        default="event_id",
    )
    parser.add_argument(
        "--profile-col",
        default="persistent_profile_id",
    )
    parser.add_argument(
        "--vocalization-value",
        default="Barking",
    )
    return parser.parse_args()


def bbox_iou(a: Iterable[float], b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def parse_cvat_xml(
    xml_path: Path,
    vocalization_value: str,
) -> pd.DataFrame:
    root = ET.parse(xml_path).getroot()
    rows: list[dict] = []

    for track in root.findall("track"):
        track_id = str(track.attrib.get("id", ""))
        label = str(track.attrib.get("label", ""))
        source = str(track.attrib.get("source", ""))

        for box in track.findall("box"):
            outside = int(box.attrib.get("outside", "0"))
            if outside == 1:
                continue

            vocalization = None
            for attribute in box.findall("attribute"):
                if attribute.attrib.get("name") == "Vocalization":
                    vocalization = (attribute.text or "").strip()

            if vocalization != vocalization_value:
                continue

            rows.append(
                {
                    "gt_track_id": track_id,
                    "label": label,
                    "source": source,
                    "frame": int(box.attrib["frame"]),
                    "x1": float(box.attrib["xtl"]),
                    "y1": float(box.attrib["ytl"]),
                    "x2": float(box.attrib["xbr"]),
                    "y2": float(box.attrib["ybr"]),
                    "occluded": int(box.attrib.get("occluded", "0")),
                    "keyframe": int(box.attrib.get("keyframe", "0")),
                    "vocalization": vocalization,
                }
            )

    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError(
            f"No '{vocalization_value}' boxes found in {xml_path}"
        )
    return result.sort_values(["gt_track_id", "frame"]).reset_index(drop=True)


def get_video_metadata(video_path: Path) -> tuple[float, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid FPS for {video_path}: {fps}")
    return fps, total_frames


def normalize_tracks(tracks: pd.DataFrame) -> pd.DataFrame:
    required = ["track_id", "frame", "x1", "y1", "x2", "y2"]
    missing = [column for column in required if column not in tracks.columns]
    if missing:
        raise KeyError(f"tracks.csv is missing columns: {missing}")

    result = tracks.copy()
    result["track_id"] = pd.to_numeric(
        result["track_id"], errors="coerce"
    ).astype("Int64")
    result["frame"] = pd.to_numeric(
        result["frame"], errors="coerce"
    ).astype("Int64")

    for column in ["x1", "y1", "x2", "y2"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    return result.dropna(
        subset=["track_id", "frame", "x1", "y1", "x2", "y2"]
    ).copy()


def normalize_predictions(
    predictions: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    required = [
        args.prediction_track_col,
        args.prediction_start_col,
        args.prediction_end_col,
    ]
    missing = [column for column in required if column not in predictions.columns]
    if missing:
        raise KeyError(
            f"Prediction file is missing columns {missing}. "
            f"Available: {predictions.columns.tolist()}"
        )

    result = predictions.copy()
    result["pred_track_id"] = pd.to_numeric(
        result[args.prediction_track_col], errors="coerce"
    ).astype("Int64")
    result["start_time_sec_eval"] = pd.to_numeric(
        result[args.prediction_start_col], errors="coerce"
    )
    result["end_time_sec_eval"] = pd.to_numeric(
        result[args.prediction_end_col], errors="coerce"
    )

    if args.prediction_event_col in result.columns:
        result["event_id_eval"] = result[
            args.prediction_event_col
        ].astype(str)
    else:
        result["event_id_eval"] = [
            f"event_{index:06d}" for index in range(len(result))
        ]

    return result.dropna(
        subset=[
            "pred_track_id",
            "start_time_sec_eval",
            "end_time_sec_eval",
        ]
    ).reset_index(drop=True)


def profile_mapping(
    mapping_frame: pd.DataFrame,
    profile_col: str,
) -> dict[int, str]:
    if "track_id" not in mapping_frame.columns:
        raise KeyError("track-profile map is missing track_id")
    if profile_col not in mapping_frame.columns:
        raise KeyError(
            f"track-profile map is missing {profile_col}. "
            f"Available: {mapping_frame.columns.tolist()}"
        )

    return {
        int(row["track_id"]): str(row[profile_col])
        for _, row in mapping_frame.iterrows()
    }


def find_nearest_track_box(
    tracks: pd.DataFrame,
    track_id: int,
    target_frame: int,
    tolerance: int,
) -> pd.Series | None:
    subset = tracks[
        tracks["track_id"].astype(int) == int(track_id)
    ].copy()
    if subset.empty:
        return None

    subset["frame_distance"] = (
        subset["frame"].astype(int) - int(target_frame)
    ).abs()
    subset = subset[
        subset["frame_distance"] <= tolerance
    ].sort_values(["frame_distance", "frame"])

    if subset.empty:
        return None
    return subset.iloc[0]


def candidate_gt_boxes(
    gt: pd.DataFrame,
    target_frame: int,
    tolerance: int,
) -> pd.DataFrame:
    subset = gt.copy()
    subset["frame_distance"] = (
        subset["frame"].astype(int) - int(target_frame)
    ).abs()
    subset = subset[
        subset["frame_distance"] <= tolerance
    ].copy()

    if subset.empty:
        return subset

    # Keep the closest annotation frame for each GT track.
    subset = (
        subset.sort_values(["gt_track_id", "frame_distance", "frame"])
        .groupby("gt_track_id", as_index=False)
        .first()
    )
    return subset


def hungarian_profile_alignment(
    events: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str]]:
    usable = events[
        events["localized_correct"] == True  # noqa: E712
    ].copy()

    if usable.empty:
        return pd.DataFrame(), {}

    profiles = sorted(usable["persistent_profile_id"].astype(str).unique())
    gt_ids = sorted(usable["matched_gt_track_id"].astype(str).unique())

    counts = np.zeros((len(profiles), len(gt_ids)), dtype=float)

    for profile_index, profile_id in enumerate(profiles):
        for gt_index, gt_id in enumerate(gt_ids):
            counts[profile_index, gt_index] = (
                (
                    usable["persistent_profile_id"].astype(str)
                    == profile_id
                )
                & (
                    usable["matched_gt_track_id"].astype(str)
                    == gt_id
                )
            ).sum()

    size = max(counts.shape)
    padded = np.zeros((size, size), dtype=float)
    padded[: counts.shape[0], : counts.shape[1]] = counts

    row_indices, column_indices = linear_sum_assignment(-padded)

    mapping: dict[str, str] = {}
    rows = []

    for row_index, column_index in zip(
        row_indices,
        column_indices,
    ):
        if row_index >= len(profiles):
            continue

        profile_id = profiles[row_index]

        if (
            column_index < len(gt_ids)
            and counts[row_index, column_index] > 0
        ):
            gt_id = gt_ids[column_index]
            score = int(counts[row_index, column_index])
        else:
            gt_id = "__unmapped__"
            score = 0

        mapping[profile_id] = gt_id
        rows.append(
            {
                "persistent_profile_id": profile_id,
                "mapped_gt_track_id": gt_id,
                "matched_event_count": score,
            }
        )

    return pd.DataFrame(rows), mapping


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else float("nan")


def main() -> None:
    args = parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fps, total_frames = get_video_metadata(args.video)
    gt = parse_cvat_xml(
        args.annotations_xml,
        args.vocalization_value,
    )
    tracks = normalize_tracks(pd.read_csv(args.tracks))
    predictions = normalize_predictions(
        pd.read_csv(args.predictions),
        args,
    )
    track_to_profile = profile_mapping(
        pd.read_csv(args.track_profile_map),
        args.profile_col,
    )

    event_rows: list[dict] = []

    for prediction in predictions.itertuples(index=False):
        start_frame = max(
            0,
            int(math.floor(prediction.start_time_sec_eval * fps)),
        )
        end_frame = min(
            total_frames - 1,
            int(math.ceil(prediction.end_time_sec_eval * fps)),
        )
        midpoint_frame = int(round((start_frame + end_frame) / 2.0))

        # Evaluate several frames inside the bark interval and keep the best.
        frames_to_check = sorted(
            {
                start_frame,
                midpoint_frame,
                end_frame,
            }
        )

        best: dict | None = None

        for target_frame in frames_to_check:
            pred_box = find_nearest_track_box(
                tracks,
                int(prediction.pred_track_id),
                target_frame,
                args.frame_tolerance,
            )
            if pred_box is None:
                continue

            gt_candidates = candidate_gt_boxes(
                gt,
                target_frame,
                args.frame_tolerance,
            )
            if gt_candidates.empty:
                continue

            for _, gt_box in gt_candidates.iterrows():
                iou = bbox_iou(
                    [
                        pred_box["x1"],
                        pred_box["y1"],
                        pred_box["x2"],
                        pred_box["y2"],
                    ],
                    [
                        gt_box["x1"],
                        gt_box["y1"],
                        gt_box["x2"],
                        gt_box["y2"],
                    ],
                )

                candidate = {
                    "evaluation_frame": int(target_frame),
                    "pred_box_frame": int(pred_box["frame"]),
                    "gt_box_frame": int(gt_box["frame"]),
                    "matched_gt_track_id": str(
                        gt_box["gt_track_id"]
                    ),
                    "spatial_iou": float(iou),
                    "pred_x1": float(pred_box["x1"]),
                    "pred_y1": float(pred_box["y1"]),
                    "pred_x2": float(pred_box["x2"]),
                    "pred_y2": float(pred_box["y2"]),
                    "gt_x1": float(gt_box["x1"]),
                    "gt_y1": float(gt_box["y1"]),
                    "gt_x2": float(gt_box["x2"]),
                    "gt_y2": float(gt_box["y2"]),
                }

                if best is None or candidate["spatial_iou"] > best["spatial_iou"]:
                    best = candidate

        persistent_profile_id = track_to_profile.get(
            int(prediction.pred_track_id),
            "__unmapped__",
        )

        row = {
            "event_id": str(prediction.event_id_eval),
            "start_time_sec": float(
                prediction.start_time_sec_eval
            ),
            "end_time_sec": float(
                prediction.end_time_sec_eval
            ),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "predicted_track_id": int(
                prediction.pred_track_id
            ),
            "persistent_profile_id": persistent_profile_id,
        }

        if best is None:
            row.update(
                {
                    "evaluation_frame": np.nan,
                    "pred_box_frame": np.nan,
                    "gt_box_frame": np.nan,
                    "matched_gt_track_id": "__none__",
                    "spatial_iou": 0.0,
                    "localized_correct": False,
                    "failure_reason": (
                        "missing_predicted_or_ground_truth_box"
                    ),
                }
            )
        else:
            row.update(best)
            row["localized_correct"] = (
                best["spatial_iou"]
                >= args.spatial_iou_threshold
            )
            row["failure_reason"] = (
                ""
                if row["localized_correct"]
                else "spatial_iou_below_threshold"
            )

        event_rows.append(row)

    events = pd.DataFrame(event_rows)

    identity_mapping, mapping = hungarian_profile_alignment(events)

    events["mapped_gt_track_id"] = (
        events["persistent_profile_id"]
        .map(mapping)
        .fillna("__unmapped__")
    )
    events["profile_identity_correct"] = (
        events["localized_correct"]
        & (
            events["mapped_gt_track_id"].astype(str)
            == events["matched_gt_track_id"].astype(str)
        )
    )

    total_events = len(events)
    localized_correct = int(events["localized_correct"].sum())
    identity_correct = int(events["profile_identity_correct"].sum())

    # Profile purity from correctly localized bark events.
    usable = events[events["localized_correct"] == True].copy()  # noqa: E712
    profile_purity_rows = []

    for profile_id, group in usable.groupby("persistent_profile_id"):
        counts = group["matched_gt_track_id"].value_counts()
        majority_gt = str(counts.index[0])
        majority_count = int(counts.iloc[0])

        profile_purity_rows.append(
            {
                "persistent_profile_id": str(profile_id),
                "matched_events": len(group),
                "unique_gt_track_ids": int(
                    group["matched_gt_track_id"].nunique()
                ),
                "majority_gt_track_id": majority_gt,
                "majority_event_count": majority_count,
                "profile_purity": safe_divide(
                    majority_count,
                    len(group),
                ),
                "merge_error_suspect": (
                    group["matched_gt_track_id"].nunique() > 1
                ),
            }
        )

    profile_purity = pd.DataFrame(profile_purity_rows)

    # GT-track fragmentation across predicted profiles.
    fragmentation_rows = []
    for gt_track_id, group in usable.groupby("matched_gt_track_id"):
        fragmentation_rows.append(
            {
                "gt_track_id": str(gt_track_id),
                "matched_events": len(group),
                "unique_persistent_profiles": int(
                    group["persistent_profile_id"].nunique()
                ),
                "persistent_profiles": ",".join(
                    sorted(
                        group[
                            "persistent_profile_id"
                        ].astype(str).unique()
                    )
                ),
                "fragmentation_suspect": (
                    group["persistent_profile_id"].nunique() > 1
                ),
            }
        )

    fragmentation = pd.DataFrame(fragmentation_rows)

    cvat_summary = (
        gt.groupby("gt_track_id")
        .agg(
            first_frame=("frame", "min"),
            last_frame=("frame", "max"),
            annotated_frames=("frame", "nunique"),
            mean_box_width=(
                "x2",
                lambda values: float(np.nan),
            ),
        )
        .reset_index()
    )

    # Replace placeholder mean width with direct summary.
    width_summary = (
        gt.assign(width=gt["x2"] - gt["x1"])
        .groupby("gt_track_id")["width"]
        .mean()
    )
    cvat_summary["mean_box_width"] = cvat_summary[
        "gt_track_id"
    ].map(width_summary)
    cvat_summary["duration_frames"] = (
        cvat_summary["last_frame"]
        - cvat_summary["first_frame"]
        + 1
    )
    cvat_summary["annotation_density"] = (
        cvat_summary["annotated_frames"]
        / cvat_summary["duration_frames"]
    )

    aggregate = pd.DataFrame(
        [
            {
                "events_evaluated": total_events,
                "localized_correct_events": localized_correct,
                "active_barker_localization_accuracy": safe_divide(
                    localized_correct,
                    total_events,
                ),
                "mean_spatial_iou": (
                    float(events["spatial_iou"].mean())
                    if not events.empty
                    else float("nan")
                ),
                "profile_identity_correct_events": identity_correct,
                "profile_identity_accuracy": safe_divide(
                    identity_correct,
                    total_events,
                ),
                "identity_accuracy_given_localization": safe_divide(
                    identity_correct,
                    localized_correct,
                ),
                "predicted_persistent_profiles": int(
                    events["persistent_profile_id"].nunique()
                ),
                "matched_gt_cvat_tracks": int(
                    usable["matched_gt_track_id"].nunique()
                ),
                "spatial_iou_threshold": (
                    args.spatial_iou_threshold
                ),
                "fps": fps,
            }
        ]
    )

    events.to_csv(
        output_dir / "bark_event_localization.csv",
        index=False,
    )
    identity_mapping.to_csv(
        output_dir / "profile_identity_mapping.csv",
        index=False,
    )
    profile_purity.to_csv(
        output_dir / "profile_purity.csv",
        index=False,
    )
    fragmentation.to_csv(
        output_dir / "gt_track_fragmentation.csv",
        index=False,
    )
    cvat_summary.to_csv(
        output_dir / "cvat_track_summary.csv",
        index=False,
    )
    aggregate.to_csv(
        output_dir / "aggregate_metrics.csv",
        index=False,
    )

    result = aggregate.iloc[0]

    summary = f"""CVAT ACTIVE-BARKER AND PROFILE EVALUATION

Video
- FPS: {fps:.6f}
- Events evaluated: {total_events}
- CVAT barking tracks: {gt['gt_track_id'].nunique()}

Active-barker localization
- Spatial IoU threshold: {args.spatial_iou_threshold:.2f}
- Correctly localized events: {localized_correct}/{total_events}
- Localization accuracy: {100 * result['active_barker_localization_accuracy']:.2f}%
- Mean spatial IoU: {result['mean_spatial_iou']:.4f}

Persistent-profile alignment
- Correct profile identities: {identity_correct}/{total_events}
- Profile identity accuracy: {100 * result['profile_identity_accuracy']:.2f}%
- Identity accuracy given correct localization: {100 * result['identity_accuracy_given_localization']:.2f}%

Important limitation
CVAT track IDs are valid identity labels only if the same track ID was maintained
for the same physical dog across the full video. If each bark event has a new
CVAT track ID, report localization accuracy but do not interpret profile purity
or fragmentation as true dog-identity metrics.
"""
    (output_dir / "evaluation_summary.txt").write_text(
        summary,
        encoding="utf-8",
    )

    print("\n" + summary)
    print("Saved:", output_dir)


if __name__ == "__main__":
    main()
