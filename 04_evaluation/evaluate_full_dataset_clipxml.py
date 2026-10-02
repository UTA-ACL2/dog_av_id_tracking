#!/usr/bin/env python3
"""
Evaluate completed multimodal runs using each bark event's clip-level
CVAT XML rather than the top-level full-video XML.

Why this version exists
-----------------------
The bark timestamps were generated from numbered clip folders such as:

    <video_id>/00/annotations.xml
    <video_id>/01/annotations.xml

Those XML files use clip-local frame numbers. The previous evaluator compared
full-video timestamps against the top-level XML and therefore incorrectly
classified most bark events as offscreen.

This evaluator maps every event ID (for example gt_bark_006) to folder 06,
then compares boxes at matching relative positions within:

    full-video bark interval <-> clip-local annotated interval

It does not use manual whole-video dog counts and does not claim that the
number of produced profiles is correct.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate outputs using clip-level CVAT annotations."
    )
    parser.add_argument("--annotations-root", type=Path, required=True)
    parser.add_argument("--pipeline-output-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--spatial-iou-threshold", type=float, default=0.30)
    parser.add_argument("--max-track-gap-sec", type=float, default=1.0)
    parser.add_argument("--samples-per-event", type=int, default=15)
    return parser.parse_args()


def pick(df: pd.DataFrame, names: list[str], required: bool = True) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    if required:
        raise KeyError(f"Missing one of {names}; columns={list(df.columns)}")
    return None


def safe_divide(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0 else 0.0


def normalize_state(value: str | None) -> str:
    value = (value or "").strip().lower().replace(" ", "_").replace("-", "_")
    if value in {"bark", "barking"}:
        return "barking"
    if value in {"silent", "not_barking", "notbarking"}:
        return "not_barking"
    return value


def parse_cvat(xml_path: Path) -> list[dict]:
    root = ET.parse(xml_path).getroot()
    tracks: list[dict] = []

    for track_element in root.findall("track"):
        rows: list[tuple] = []
        for box in track_element.findall("box"):
            if box.attrib.get("outside", "0") == "1":
                continue

            state = ""
            for attribute in box.findall("attribute"):
                if attribute.attrib.get("name", "").strip().lower() == "vocalization":
                    state = normalize_state(attribute.text)
                    break

            rows.append(
                (
                    int(box.attrib["frame"]),
                    float(box.attrib["xtl"]),
                    float(box.attrib["ytl"]),
                    float(box.attrib["xbr"]),
                    float(box.attrib["ybr"]),
                    state,
                )
            )

        if rows:
            rows.sort(key=lambda row: row[0])
            tracks.append(
                {
                    "id": int(track_element.attrib.get("id", len(tracks))),
                    "rows": rows,
                }
            )

    return tracks


def cvat_box_and_state_at(track: dict, frame: int) -> tuple[tuple[float, float, float, float] | None, str]:
    rows = track["rows"]
    frames = np.asarray([row[0] for row in rows], dtype=np.int64)
    position = int(np.searchsorted(frames, frame))

    if position < len(rows) and int(frames[position]) == frame:
        row = rows[position]
        return (row[1], row[2], row[3], row[4]), row[5]

    if position == 0 or position >= len(rows):
        return None, ""

    left = rows[position - 1]
    right = rows[position]
    if not (left[0] < frame < right[0]):
        return None, ""

    gap = right[0] - left[0]
    if gap <= 0:
        return None, ""

    alpha = (frame - left[0]) / gap
    interpolated = tuple(
        float(left[index] * (1.0 - alpha) + right[index] * alpha)
        for index in range(1, 5)
    )
    state = left[5] if left[5] == right[5] else ""
    return interpolated, state


def clip_frame_bounds(tracks: list[dict]) -> tuple[int, int] | None:
    frames = [row[0] for track in tracks for row in track["rows"]]
    if not frames:
        return None
    return min(frames), max(frames)


def resolve_event_clip_xml(video_dir: Path, event_id: str) -> tuple[Path | None, str]:
    match = re.search(r"(\d+)(?!.*\d)", str(event_id))
    if match is None:
        return None, "event_id_has_no_numeric_suffix"

    index = int(match.group(1))
    candidates = [
        video_dir / f"{index:02d}" / "annotations.xml",
        video_dir / str(index) / "annotations.xml",
        video_dir / f"{index:03d}" / "annotations.xml",
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate, ""

    return None, f"clip_xml_not_found_for_index_{index}"


def normalize_tracks(path: Path) -> pd.DataFrame:
    source = pd.read_csv(path)
    output = pd.DataFrame()
    output["track_id"] = pd.to_numeric(
        source[pick(source, ["track_id", "track", "id"])], errors="raise"
    ).astype(int)
    output["frame"] = pd.to_numeric(
        source[pick(source, ["frame", "frame_idx", "frame_id"])], errors="raise"
    ).round().astype(int)

    for canonical, names in {
        "x1": ["x1", "xtl", "left"],
        "y1": ["y1", "ytl", "top"],
        "x2": ["x2", "xbr", "right"],
        "y2": ["y2", "ybr", "bottom"],
    }.items():
        output[canonical] = pd.to_numeric(source[pick(source, names)], errors="raise").astype(float)

    if "time_sec" in source.columns:
        output["time_sec"] = pd.to_numeric(source["time_sec"], errors="coerce")

    return (
        output
        .dropna(subset=["track_id", "frame", "x1", "y1", "x2", "y2"])
        .sort_values(["track_id", "frame"])
        .drop_duplicates(["track_id", "frame"], keep="last")
        .reset_index(drop=True)
    )


def infer_fps(tracks: pd.DataFrame) -> float:
    if "time_sec" in tracks.columns:
        valid = tracks["time_sec"] > 0
        estimates = tracks.loc[valid, "frame"] / tracks.loc[valid, "time_sec"]
        estimates = estimates[np.isfinite(estimates) & (estimates > 1)]
        if len(estimates):
            return float(np.median(estimates))
    return 30.0


def interpolate_track_box(
    rows: pd.DataFrame,
    frame: int,
    max_gap_frames: int,
) -> tuple[float, float, float, float] | None:
    frames = rows["frame"].to_numpy(dtype=np.int64)
    boxes = rows[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
    if not len(frames):
        return None

    position = int(np.searchsorted(frames, frame))
    if position < len(frames) and int(frames[position]) == frame:
        return tuple(float(value) for value in boxes[position])

    left_index = position - 1 if position > 0 else None
    right_index = position if position < len(frames) else None

    if left_index is not None and right_index is not None:
        left_frame = int(frames[left_index])
        right_frame = int(frames[right_index])
        gap = right_frame - left_frame
        if left_frame < frame < right_frame and 0 < gap <= 2 * max_gap_frames:
            alpha = (frame - left_frame) / gap
            interpolated = boxes[left_index] * (1.0 - alpha) + boxes[right_index] * alpha
            return tuple(float(value) for value in interpolated)

    candidates = [index for index in (left_index, right_index) if index is not None]
    if not candidates:
        return None

    nearest = min(candidates, key=lambda index: abs(int(frames[index]) - frame))
    if abs(int(frames[nearest]) - frame) > max_gap_frames:
        return None
    return tuple(float(value) for value in boxes[nearest])


def normalize_predictions(path: Path) -> pd.DataFrame:
    source = pd.read_csv(path)
    output = pd.DataFrame()
    output["event_id"] = source[pick(source, ["event_id", "barkseq_id", "bark_id"])].astype(str)
    output["start"] = pd.to_numeric(
        source[pick(source, ["start_time_sec", "start_time", "start_sec", "start"])],
        errors="raise",
    )
    output["end"] = pd.to_numeric(
        source[pick(source, ["end_time_sec", "end_time", "end_sec", "end"])],
        errors="raise",
    )
    track_column = pick(
        source,
        ["predicted_track_id", "track_id", "predicted_id"],
        required=False,
    )
    status_column = pick(source, ["status", "prediction_status"], required=False)
    output["predicted_track_id"] = (
        pd.to_numeric(source[track_column], errors="coerce")
        if track_column is not None
        else np.nan
    )
    output["status"] = source[status_column].astype(str) if status_column else "predicted"
    return output


def sample_event_positions(start: float, end: float, fps: float, count: int) -> list[tuple[int, float]]:
    start_frame = int(math.floor(start * fps))
    end_frame = int(math.ceil(end * fps))
    if end_frame <= start_frame:
        return [(start_frame, 0.5)]

    sample_count = min(count, end_frame - start_frame + 1)
    full_frames = np.linspace(start_frame, end_frame, num=sample_count)
    output: list[tuple[int, float]] = []
    denominator = max(1, end_frame - start_frame)
    for value in full_frames:
        frame = int(round(value))
        alpha = min(1.0, max(0.0, (frame - start_frame) / denominator))
        output.append((frame, alpha))
    return list(dict.fromkeys(output))


def profile_count(path: Path) -> float:
    if not path.is_file():
        return float("nan")
    source = pd.read_csv(path)
    column = pick(
        source,
        ["all_dog_profile_id", "persistent_profile_id", "profile_id", "all_dog_id"],
        required=False,
    )
    return int(source[column].nunique()) if column else float("nan")


def evaluate_video(
    video_id: str,
    video_annotation_dir: Path,
    base_dir: Path,
    profile_dir: Path,
    fusion_dir: Path,
    threshold: float,
    max_gap_sec: float,
    samples: int,
) -> tuple[list[dict], dict, dict, dict]:
    predictions = normalize_predictions(base_dir / "predictions.csv")
    tracks = normalize_tracks(base_dir / "tracks.csv")
    fps = infer_fps(tracks)
    max_gap_frames = max(1, int(round(max_gap_sec * fps)))
    grouped_tracks = {
        int(track_id): group.reset_index(drop=True)
        for track_id, group in tracks.groupby("track_id")
    }

    event_rows: list[dict] = []

    for _, event in predictions.iterrows():
        event_id = str(event.event_id)
        clip_xml, clip_error = resolve_event_clip_xml(video_annotation_dir, event_id)
        status = str(event.status).strip().lower()
        predicted_track = event.predicted_track_id
        predicted_visible = (
            pd.notna(predicted_track)
            and status not in {"outside_or_offscreen", "offscreen", "outside"}
        )

        if clip_xml is None:
            event_rows.append(
                {
                    "video_id": video_id,
                    "event_id": event_id,
                    "clip_xml": "",
                    "evaluation_status": clip_error,
                    "start_time_sec": float(event.start),
                    "end_time_sec": float(event.end),
                    "prediction_status": event.status,
                    "predicted_track_id": int(predicted_track) if pd.notna(predicted_track) else "",
                    "gt_visible_barker": np.nan,
                    "gt_offscreen": np.nan,
                    "predicted_visible_barker": int(predicted_visible),
                    "predicted_offscreen": int(not predicted_visible),
                    "mean_spatial_iou": np.nan,
                    "max_spatial_iou": np.nan,
                    "localization_correct": np.nan,
                    "matched_gt_track_id": "",
                    "sampled_frames": 0,
                    "gt_visible_sampled_frames": 0,
                    "evaluable_sampled_frames": 0,
                }
            )
            continue

        cvat_tracks = parse_cvat(clip_xml)
        bounds = clip_frame_bounds(cvat_tracks)
        if bounds is None:
            clip_start, clip_end = 0, 0
        else:
            clip_start, clip_end = bounds

        sampled_positions = sample_event_positions(
            float(event.start), float(event.end), fps, samples
        )

        gt_visible_frames = 0
        evaluable_frames = 0
        ious: list[float] = []
        matched_ids: list[int] = []

        for full_frame, alpha in sampled_positions:
            clip_frame = int(round(clip_start + alpha * max(0, clip_end - clip_start)))
            gt_boxes: list[tuple[int, tuple[float, float, float, float]]] = []

            for gt_track in cvat_tracks:
                gt_box, gt_state = cvat_box_and_state_at(gt_track, clip_frame)
                if gt_box is not None and gt_state == "barking":
                    gt_boxes.append((int(gt_track["id"]), gt_box))

            if gt_boxes:
                gt_visible_frames += 1

            if not (predicted_visible and gt_boxes):
                continue

            pipeline_rows = grouped_tracks.get(int(predicted_track))
            if pipeline_rows is None:
                continue

            predicted_box = interpolate_track_box(
                pipeline_rows, full_frame, max_gap_frames
            )
            if predicted_box is None:
                continue

            best_gt_id, best_iou = max(
                (
                    (gt_track_id, box_iou(predicted_box, gt_box))
                    for gt_track_id, gt_box in gt_boxes
                ),
                key=lambda item: item[1],
            )
            evaluable_frames += 1
            ious.append(float(best_iou))
            matched_ids.append(int(best_gt_id))

        gt_visible = gt_visible_frames > 0
        mean_iou = float(np.mean(ious)) if ious else float("nan")
        localization_correct = bool(
            gt_visible
            and predicted_visible
            and ious
            and mean_iou >= threshold
        )
        dominant_gt_id = (
            int(pd.Series(matched_ids).mode().iloc[0]) if matched_ids else ""
        )

        event_rows.append(
            {
                "video_id": video_id,
                "event_id": event_id,
                "clip_xml": str(clip_xml),
                "evaluation_status": "evaluated",
                "start_time_sec": float(event.start),
                "end_time_sec": float(event.end),
                "prediction_status": event.status,
                "predicted_track_id": int(predicted_track) if pd.notna(predicted_track) else "",
                "gt_visible_barker": int(gt_visible),
                "gt_offscreen": int(not gt_visible),
                "predicted_visible_barker": int(predicted_visible),
                "predicted_offscreen": int(not predicted_visible),
                "mean_spatial_iou": mean_iou,
                "max_spatial_iou": float(np.max(ious)) if ious else float("nan"),
                "localization_correct": int(localization_correct),
                "matched_gt_track_id": dominant_gt_id,
                "sampled_frames": len(sampled_positions),
                "gt_visible_sampled_frames": gt_visible_frames,
                "evaluable_sampled_frames": evaluable_frames,
            }
        )

    events = pd.DataFrame(event_rows)
    evaluated = events[events["evaluation_status"] == "evaluated"].copy()
    visible = evaluated[evaluated["gt_visible_barker"] == 1]
    visible_predictions = visible[visible["predicted_visible_barker"] == 1]

    tp = int(((evaluated.gt_visible_barker == 1) & (evaluated.predicted_visible_barker == 1)).sum())
    fp = int(((evaluated.gt_visible_barker == 0) & (evaluated.predicted_visible_barker == 1)).sum())
    fn = int(((evaluated.gt_visible_barker == 1) & (evaluated.predicted_visible_barker == 0)).sum())
    tn = int(((evaluated.gt_visible_barker == 0) & (evaluated.predicted_visible_barker == 0)).sum())
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = (
        safe_divide(2 * precision * recall, precision + recall)
        if np.isfinite(precision) and np.isfinite(recall)
        else float("nan")
    )

    video_row = {
        "video_id": video_id,
        "events_total": len(events),
        "events_evaluated": len(evaluated),
        "events_missing_clip_xml": int((events.evaluation_status != "evaluated").sum()),
        "gt_visible_events": len(visible),
        "gt_offscreen_events": int(evaluated.gt_offscreen.sum()),
        "correctly_localized_visible_events": int(visible.localization_correct.sum()),
        "localization_accuracy_all_visible": safe_divide(
            visible.localization_correct.sum(), len(visible)
        ),
        "localization_accuracy_given_visible_prediction": safe_divide(
            visible_predictions.localization_correct.sum(), len(visible_predictions)
        ),
        "mean_spatial_iou": (
            float(visible_predictions.mean_spatial_iou.mean())
            if len(visible_predictions)
            else float("nan")
        ),
        "visible_precision": precision,
        "visible_recall": recall,
        "visible_f1": f1,
        "visible_offscreen_accuracy": safe_divide(tp + tn, tp + fp + fn + tn),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }

    temporary_tracks = int(tracks.track_id.nunique())
    persistent_profiles = profile_count(profile_dir / "track_to_all_dog_profile.csv")
    merge_row = {
        "video_id": video_id,
        "temporary_tracks": temporary_tracks,
        "persistent_profiles_produced": persistent_profiles,
        "fragments_merged": (
            temporary_tracks - persistent_profiles
            if np.isfinite(persistent_profiles)
            else float("nan")
        ),
    }

    fusion_row = {"video_id": video_id}
    fusion_summary = fusion_dir / "pipeline_summary_v2.csv"
    if fusion_summary.is_file():
        source = pd.read_csv(fusion_summary)
        if not source.empty:
            fusion_row.update(source.iloc[0].to_dict())

    return event_rows, video_row, merge_row, fusion_row


def main() -> None:
    args = parse_args()
    annotations_root = args.annotations_root.expanduser().resolve()
    pipeline_root = args.pipeline_output_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    video_dirs = sorted(
        [
            path
            for path in annotations_root.iterdir()
            if path.is_dir() and path.name.isdigit()
        ],
        key=lambda path: int(path.name),
    )

    event_rows: list[dict] = []
    video_rows: list[dict] = []
    merge_rows: list[dict] = []
    fusion_rows: list[dict] = []
    failures: list[dict] = []

    for video_dir in video_dirs:
        video_id = video_dir.name
        base_dir = pipeline_root / f"{video_id}_base"
        profile_dir = pipeline_root / f"{video_id}_profiles"
        fusion_dir = pipeline_root / f"{video_id}_fusion"

        required = [base_dir / "predictions.csv", base_dir / "tracks.csv"]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            failures.append(
                {
                    "video_id": video_id,
                    "error": "missing_required_files",
                    "details": ";".join(missing),
                }
            )
            print(f"SKIP {video_id}: missing required outputs")
            continue

        try:
            current_events, video_row, merge_row, fusion_row = evaluate_video(
                video_id,
                video_dir,
                base_dir,
                profile_dir,
                fusion_dir,
                args.spatial_iou_threshold,
                args.max_track_gap_sec,
                args.samples_per_event,
            )
            event_rows.extend(current_events)
            video_rows.append(video_row)
            merge_rows.append(merge_row)
            fusion_rows.append(fusion_row)
            print(
                f"{video_id}: "
                f"{video_row['correctly_localized_visible_events']}/"
                f"{video_row['gt_visible_events']} visible barks correct; "
                f"{video_row['events_missing_clip_xml']} missing clip XML"
            )
        except Exception as error:
            failures.append(
                {
                    "video_id": video_id,
                    "error": type(error).__name__,
                    "details": str(error),
                }
            )
            print(f"FAILED {video_id}: {type(error).__name__}: {error}")

    events = pd.DataFrame(event_rows)
    videos = pd.DataFrame(video_rows)
    merges = pd.DataFrame(merge_rows)
    fusion = pd.DataFrame(fusion_rows)

    events.to_csv(output_dir / "per_event_metrics.csv", index=False)
    videos.to_csv(output_dir / "per_video_metrics.csv", index=False)
    merges.to_csv(output_dir / "track_merge_statistics.csv", index=False)
    fusion.to_csv(output_dir / "fusion_statistics.csv", index=False)
    pd.DataFrame(failures).to_csv(output_dir / "failures.csv", index=False)

    if events.empty:
        raise SystemExit("No events evaluated")

    evaluated = events[events.evaluation_status == "evaluated"].copy()
    visible = evaluated[evaluated.gt_visible_barker == 1]
    visible_predictions = visible[visible.predicted_visible_barker == 1]

    tp = int(((evaluated.gt_visible_barker == 1) & (evaluated.predicted_visible_barker == 1)).sum())
    fp = int(((evaluated.gt_visible_barker == 0) & (evaluated.predicted_visible_barker == 1)).sum())
    fn = int(((evaluated.gt_visible_barker == 1) & (evaluated.predicted_visible_barker == 0)).sum())
    tn = int(((evaluated.gt_visible_barker == 0) & (evaluated.predicted_visible_barker == 0)).sum())
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = (
        safe_divide(2 * precision * recall, precision + recall)
        if np.isfinite(precision) and np.isfinite(recall)
        else float("nan")
    )

    summary = {
        "videos_discovered": len(video_dirs),
        "videos_evaluated": int(videos.video_id.nunique()),
        "videos_failed_or_skipped": len(failures),
        "bark_events_total": len(events),
        "bark_events_evaluated": len(evaluated),
        "bark_events_missing_clip_xml": int((events.evaluation_status != "evaluated").sum()),
        "gt_visible_bark_events": len(visible),
        "gt_offscreen_bark_events": int(evaluated.gt_offscreen.sum()),
        "correctly_localized_visible_events": int(visible.localization_correct.sum()),
        "micro_localization_accuracy_all_visible": safe_divide(
            visible.localization_correct.sum(), len(visible)
        ),
        "micro_localization_accuracy_given_visible_prediction": safe_divide(
            visible_predictions.localization_correct.sum(), len(visible_predictions)
        ),
        "macro_localization_accuracy_all_visible": float(
            videos.localization_accuracy_all_visible.mean()
        ),
        "mean_spatial_iou": (
            float(visible_predictions.mean_spatial_iou.mean())
            if len(visible_predictions)
            else float("nan")
        ),
        "visible_detection_precision": precision,
        "visible_detection_recall": recall,
        "visible_detection_f1": f1,
        "visible_offscreen_classification_accuracy": safe_divide(
            tp + tn, tp + fp + fn + tn
        ),
        "visible_tp": tp,
        "visible_fp": fp,
        "visible_fn": fn,
        "visible_tn": tn,
        "temporary_tracks_total": int(merges.temporary_tracks.sum()),
        "persistent_profiles_produced_total": float(
            merges.persistent_profiles_produced.sum()
        ),
        "fragments_merged_total": float(merges.fragments_merged.sum()),
    }

    if not fusion.empty:
        for column in fusion.columns:
            if column != "video_id":
                fusion[column] = pd.to_numeric(fusion[column], errors="coerce")

        if "bark_events" in fusion:
            summary["fusion_bark_events"] = float(fusion.bark_events.sum())
        if "assigned_bark_events" in fusion:
            summary["fusion_assigned_bark_events"] = float(
                fusion.assigned_bark_events.sum()
            )
            summary["fusion_assignment_rate"] = safe_divide(
                fusion.assigned_bark_events.sum(), fusion.bark_events.sum()
            )
        if "audio_visual_conflicts" in fusion:
            summary["audio_visual_conflicts_total"] = float(
                fusion.audio_visual_conflicts.sum()
            )
            summary["audio_visual_conflict_rate"] = safe_divide(
                fusion.audio_visual_conflicts.sum(),
                fusion.assigned_bark_events.sum(),
            )
        if "offscreen_audio_only_profiles" in fusion:
            summary["offscreen_audio_only_profiles_total"] = float(
                fusion.offscreen_audio_only_profiles.sum()
            )

    pd.DataFrame(
        [{"metric": key, "value": value} for key, value in summary.items()]
    ).to_csv(output_dir / "overall_summary.csv", index=False)
    (output_dir / "overall_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\nOVERALL SUMMARY")
    for key, value in summary.items():
        if isinstance(value, float) and any(
            token in key
            for token in ["accuracy", "precision", "recall", "f1", "rate"]
        ):
            print(f"{key}: {value * 100:.2f}%")
        else:
            print(f"{key}: {value}")
    print("\nSaved:", output_dir)


if __name__ == "__main__":
    main()
