#!/usr/bin/env python3
"""Build persistent visual profiles for all tracked dogs (v2.4).

Key fixes over v2:
1. Uses actual observed frames rather than each track's broad first-last span.
2. Allows duplicate track IDs that coexist on the same dog to merge when their
   boxes strongly overlap.
3. Prevents merging tracks that coexist with low spatial overlap.
4. Writes pairwise diagnostics so thresholds can be inspected.

Outputs:
- track_to_all_dog_profile.csv
- all_dog_visual_profiles.json
- track_visual_embeddings.npz
- visual_track_merges.csv
- visual_track_pair_diagnostics.csv
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import torchreid
import torchvision.transforms as T


@dataclass(frozen=True)
class TrackInfo:
    track_id: int
    first_frame: int
    last_frame: int
    first_time: float
    last_time: float
    rows: int
    mean_confidence: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create all-dog persistent visual profiles."
    )
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--tracks", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--appearance-model", default="osnet_x0_25")
    p.add_argument("--samples-per-track", type=int, default=12)
    p.add_argument("--min-track-rows", type=int, default=2)

    # Non-overlapping / fragmented-track ReID.
    p.add_argument(
        "--visual-similarity-threshold",
        type=float,
        default=0.65,
    )
    p.add_argument("--max-reid-gap-sec", type=float, default=120.0)

    # Duplicate tracks that coexist on the same physical dog.
    p.add_argument(
        "--duplicate-iou-threshold",
        type=float,
        default=0.60,
    )
    p.add_argument(
        "--duplicate-appearance-threshold",
        type=float,
        default=0.45,
    )
    p.add_argument(
        "--part-duplicate-appearance-threshold",
        type=float,
        default=0.88,
        help=(
            "Appearance threshold for simultaneous head/body-part duplicate "
            "detections belonging to one physical dog."
        ),
    )
    p.add_argument(
        "--part-duplicate-max-normalized-gap",
        type=float,
        default=0.20,
        help=(
            "Maximum median edge-to-edge box gap divided by the smaller box "
            "diagonal for simultaneous body-part duplicate merging."
        ),
    )
    p.add_argument(
        "--part-duplicate-min-iou",
        type=float,
        default=0.05,
        help=(
            "Minimum median IoU for simultaneous part detections. A pair may "
            "also pass through the normalized edge-gap criterion."
        ),
    )
    p.add_argument(
        "--min-common-frames",
        type=int,
        default=2,
    )

    # Tracks that coexist with low IoU are treated as different dogs.
    p.add_argument(
        "--different-dog-iou-threshold",
        type=float,
        default=0.20,
    )
    p.add_argument(
        "--substantial-overlap-frames",
        type=int,
        default=3,
    )
    p.add_argument(
        "--static-min-duration-sec",
        type=float,
        default=5.0,
    )
    p.add_argument(
        "--static-max-normalized-motion",
        type=float,
        default=0.15,
        help=(
            "Suppress long tracks whose total center movement divided by "
            "median box diagonal is below this value."
        ),
    )
    p.add_argument(
        "--sparse-max-density",
        type=float,
        default=0.02,
    )
    p.add_argument(
        "--sparse-max-rows",
        type=int,
        default=20,
    )
    p.add_argument(
        "--intro-outro-window-sec",
        type=float,
        default=8.0,
        help="Window at the beginning/end treated as possible title-card footage.",
    )
    p.add_argument(
        "--internal-motion-threshold",
        type=float,
        default=1.5,
        help=(
            "Median grayscale crop difference below which a track is nearly "
            "pixel-static. Used conservatively with intro/outro context."
        ),
    )
    p.add_argument(
        "--motion-samples-per-track",
        type=int,
        default=16,
    )
    p.add_argument(
        "--suppress-likely-images",
        action="store_true",
        help=(
            "Suppress tracks classified as likely intro/outro photos or screens. "
            "Without this flag, they remain profiles but are marked uncertain."
        ),
    )
    p.add_argument("--force", action="store_true")
    return p.parse_args()


class AppearanceEncoder:
    transform = T.Compose(
        [
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    def __init__(self, model_name: str, device: str) -> None:
        self.device = torch.device(device)
        self.model = torchreid.models.build_model(
            model_name,
            num_classes=1,
            pretrained=True,
        )
        self.model.eval().to(self.device)

    @torch.no_grad()
    def encode(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.empty((0, 512), dtype=np.float32)

        tensors = []
        for crop in crops:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensors.append(self.transform(rgb))

        batch = torch.stack(tensors).to(self.device)
        embeddings = self.model(batch)
        embeddings = embeddings / torch.linalg.vector_norm(
            embeddings,
            dim=1,
            keepdim=True,
        ).clamp(min=1e-6)

        return embeddings.cpu().numpy().astype(np.float32)


def cosine(first: np.ndarray, second: np.ndarray) -> float:
    return float(
        np.dot(first, second)
        / (
            (np.linalg.norm(first) * np.linalg.norm(second))
            + 1e-12
        )
    )


def bbox_iou(first: tuple[float, float, float, float],
             second: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection

    return intersection / union if union > 0 else 0.0



def box_geometry_metrics(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> dict[str, float]:
    """Return spatial metrics useful for duplicate body-part detections."""
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second

    aw = max(0.0, ax2 - ax1)
    ah = max(0.0, ay2 - ay1)
    bw = max(0.0, bx2 - bx1)
    bh = max(0.0, by2 - by1)

    acx = (ax1 + ax2) / 2.0
    acy = (ay1 + ay2) / 2.0
    bcx = (bx1 + bx2) / 2.0
    bcy = (by1 + by2) / 2.0

    center_distance = float(np.hypot(acx - bcx, acy - bcy))

    horizontal_gap = max(0.0, max(ax1, bx1) - min(ax2, bx2))
    vertical_gap = max(0.0, max(ay1, by1) - min(ay2, by2))
    edge_gap = float(np.hypot(horizontal_gap, vertical_gap))

    first_diag = float(np.hypot(aw, ah))
    second_diag = float(np.hypot(bw, bh))
    smaller_diag = max(1e-6, min(first_diag, second_diag))
    mean_diag = max(1e-6, (first_diag + second_diag) / 2.0)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

    first_area = aw * ah
    second_area = bw * bh
    smaller_area = max(1e-6, min(first_area, second_area))

    return {
        "edge_gap": edge_gap,
        "normalized_edge_gap": edge_gap / smaller_diag,
        "center_distance": center_distance,
        "normalized_center_distance": center_distance / mean_diag,
        "intersection_over_smaller": intersection / smaller_area,
        "area_ratio": (
            min(first_area, second_area) / max(first_area, second_area)
            if max(first_area, second_area) > 0
            else 0.0
        ),
    }


def select_rows(group: pd.DataFrame, count: int) -> pd.DataFrame:
    group = group.sort_values("frame").reset_index(drop=True)
    if len(group) <= count:
        return group

    indexes = (
        np.linspace(0, len(group) - 1, num=count)
        .round()
        .astype(int)
    )
    return group.iloc[np.unique(indexes)].copy()


def crop_from_row(
    frame: np.ndarray,
    row: pd.Series,
) -> np.ndarray | None:
    height, width = frame.shape[:2]

    x1 = max(0, min(width, int(math.floor(float(row["x1"])))))
    y1 = max(0, min(height, int(math.floor(float(row["y1"])))))
    x2 = max(0, min(width, int(math.ceil(float(row["x2"])))))
    y2 = max(0, min(height, int(math.ceil(float(row["y2"])))))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    return crop if crop.size else None


def extract_track_embeddings(
    video: Path,
    tracks: pd.DataFrame,
    encoder: AppearanceEncoder,
    samples_per_track: int,
) -> tuple[dict[int, np.ndarray], dict[int, int]]:
    selected = {
        int(track_id): select_rows(group, samples_per_track)
        for track_id, group in tracks.groupby("track_id")
    }

    requests: dict[int, list[tuple[int, pd.Series]]] = {}
    for track_id, rows in selected.items():
        for _, row in rows.iterrows():
            requests.setdefault(int(row["frame"]), []).append(
                (track_id, row)
            )

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")

    by_track: dict[int, list[np.ndarray]] = {
        track_id: [] for track_id in selected
    }

    try:
        for frame_index in sorted(requests):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue

            valid: list[tuple[int, np.ndarray]] = []
            for track_id, row in requests[frame_index]:
                crop = crop_from_row(frame, row)
                if crop is not None:
                    valid.append((track_id, crop))

            if not valid:
                continue

            embeddings = encoder.encode(
                [crop for _, crop in valid]
            )

            for (track_id, _), embedding in zip(valid, embeddings):
                if np.any(embedding):
                    by_track[track_id].append(embedding)
    finally:
        capture.release()

    centroids: dict[int, np.ndarray] = {}
    sample_counts: dict[int, int] = {}

    for track_id, embeddings in by_track.items():
        if not embeddings:
            continue

        centroid = np.mean(np.stack(embeddings), axis=0)
        norm = np.linalg.norm(centroid)
        if norm <= 0:
            continue

        centroids[track_id] = (
            centroid / norm
        ).astype(np.float32)
        sample_counts[track_id] = len(embeddings)

    return centroids, sample_counts


def nearest_frame_gap(
    left_frames: np.ndarray,
    right_frames: np.ndarray,
) -> int:
    """Return the smallest absolute frame distance efficiently."""
    left = np.sort(left_frames)
    right = np.sort(right_frames)

    i = 0
    j = 0
    best = np.iinfo(np.int64).max

    while i < len(left) and j < len(right):
        best = min(best, abs(int(left[i]) - int(right[j])))
        if left[i] < right[j]:
            i += 1
        else:
            j += 1

    return int(best) if best != np.iinfo(np.int64).max else 0



def resize_gray(crop: np.ndarray, size: tuple[int, int] = (96, 96)) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, size, interpolation=cv2.INTER_AREA).astype(np.float32)


def estimate_internal_motion(
    video: Path,
    tracks: pd.DataFrame,
    samples_per_track: int,
) -> dict[int, dict[str, float]]:
    """Estimate within-box pixel motion for every track.

    This uses actual observed track frames. Consecutive sampled crops are resized
    to a common shape and compared with mean absolute grayscale difference.
    A still image or title card should have very low internal motion, while a
    stationary live dog may still show breathing, head, fur, or camera motion.
    """
    selected: dict[int, pd.DataFrame] = {}
    for track_id, group in tracks.groupby("track_id"):
        group = group.sort_values("frame").drop_duplicates("frame")
        if len(group) > samples_per_track:
            indexes = (
                np.linspace(0, len(group) - 1, samples_per_track)
                .round()
                .astype(int)
            )
            group = group.iloc[np.unique(indexes)]
        selected[int(track_id)] = group

    requests: dict[int, list[tuple[int, pd.Series]]] = {}
    for track_id, group in selected.items():
        for _, row in group.iterrows():
            requests.setdefault(int(row["frame"]), []).append((track_id, row))

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video for motion analysis: {video}")

    crops_by_track: dict[int, list[tuple[int, np.ndarray]]] = {
        track_id: [] for track_id in selected
    }

    try:
        for frame_index in sorted(requests):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue

            for track_id, row in requests[frame_index]:
                crop = crop_from_row(frame, row)
                if crop is None:
                    continue
                crops_by_track[track_id].append(
                    (frame_index, resize_gray(crop))
                )
    finally:
        capture.release()

    output: dict[int, dict[str, float]] = {}
    for track_id, samples in crops_by_track.items():
        samples = sorted(samples, key=lambda item: item[0])
        differences: list[float] = []

        for (_, first), (_, second) in zip(samples, samples[1:]):
            differences.append(float(np.mean(np.abs(second - first))))

        output[track_id] = {
            "internal_motion_pairs": len(differences),
            "mean_internal_motion": (
                float(np.mean(differences)) if differences else float("nan")
            ),
            "median_internal_motion": (
                float(np.median(differences)) if differences else float("nan")
            ),
            "max_internal_motion": (
                float(np.max(differences)) if differences else float("nan")
            ),
        }

    return output

def main() -> None:
    args = parse_args()

    video = args.video.expanduser().resolve()
    tracks_path = args.tracks.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mapping_path = output_dir / "track_to_all_dog_profile.csv"
    if mapping_path.exists() and not args.force:
        raise FileExistsError(
            f"{mapping_path} exists; use --force to replace it"
        )

    if not video.is_file():
        raise FileNotFoundError(video)
    if not tracks_path.is_file():
        raise FileNotFoundError(tracks_path)

    tracks = pd.read_csv(tracks_path)

    required = {
        "frame",
        "time_sec",
        "track_id",
        "x1",
        "y1",
        "x2",
        "y2",
    }
    missing = required - set(tracks.columns)
    if missing:
        raise KeyError(
            f"Missing tracks columns: {sorted(missing)}"
        )

    tracks["track_id"] = pd.to_numeric(
        tracks["track_id"],
        errors="raise",
    ).astype(int)
    tracks["frame"] = pd.to_numeric(
        tracks["frame"],
        errors="raise",
    ).astype(int)

    # Keep one row per track per frame. Prefer the highest confidence row.
    if "confidence" in tracks.columns:
        tracks["confidence"] = pd.to_numeric(
            tracks["confidence"],
            errors="coerce",
        )
        tracks = tracks.sort_values(
            ["track_id", "frame", "confidence"],
            ascending=[True, True, False],
        )
    tracks = tracks.drop_duplicates(
        ["track_id", "frame"],
        keep="first",
    ).copy()

    track_info: dict[int, TrackInfo] = {}
    observed_frames: dict[int, np.ndarray] = {}
    boxes_by_track: dict[
        int,
        dict[int, tuple[float, float, float, float]],
    ] = {}
    quality_by_track: dict[int, dict[str, Any]] = {}

    for track_id, group in tracks.groupby("track_id"):
        track_id = int(track_id)
        confidence = (
            pd.to_numeric(
                group["confidence"],
                errors="coerce",
            ).mean()
            if "confidence" in group.columns
            else float("nan")
        )

        track_info[track_id] = TrackInfo(
            track_id=track_id,
            first_frame=int(group["frame"].min()),
            last_frame=int(group["frame"].max()),
            first_time=float(group["time_sec"].min()),
            last_time=float(group["time_sec"].max()),
            rows=int(len(group)),
            mean_confidence=float(confidence),
        )

        observed_frames[track_id] = np.sort(
            group["frame"].astype(int).unique()
        )

        boxes_by_track[track_id] = {
            int(row["frame"]): (
                float(row["x1"]),
                float(row["y1"]),
                float(row["x2"]),
                float(row["y2"]),
            )
            for _, row in group.iterrows()
        }

        ordered = group.sort_values("frame")
        centers = np.column_stack(
            (
                (ordered["x1"].to_numpy() + ordered["x2"].to_numpy()) / 2.0,
                (ordered["y1"].to_numpy() + ordered["y2"].to_numpy()) / 2.0,
            )
        )
        total_motion = (
            float(np.linalg.norm(np.diff(centers, axis=0), axis=1).sum())
            if len(centers) > 1
            else 0.0
        )
        widths = (ordered["x2"] - ordered["x1"]).clip(lower=0).to_numpy()
        heights = (ordered["y2"] - ordered["y1"]).clip(lower=0).to_numpy()
        median_diagonal = float(
            np.median(np.sqrt(widths * widths + heights * heights))
        )
        frame_span = max(
            1,
            int(ordered["frame"].max() - ordered["frame"].min() + 1),
        )
        density = len(ordered) / frame_span
        duration = float(
            ordered["time_sec"].max() - ordered["time_sec"].min()
        )
        normalized_motion = total_motion / max(median_diagonal, 1e-6)

        static_suspect = (
            duration >= args.static_min_duration_sec
            and normalized_motion <= args.static_max_normalized_motion
        )
        sparse_suspect = (
            density <= args.sparse_max_density
            and len(ordered) <= args.sparse_max_rows
        )

        quality_by_track[track_id] = {
            "duration_sec": duration,
            "frame_span": frame_span,
            "observation_density": density,
            "total_center_movement": total_motion,
            "median_box_diagonal": median_diagonal,
            "normalized_motion": normalized_motion,
            "static_suspect": static_suspect,
            "sparse_suspect": sparse_suspect,
        }

    eligible_track_ids = {
        track_id
        for track_id, info in track_info.items()
        if info.rows >= args.min_track_rows
    }
    eligible = tracks[
        tracks["track_id"].isin(eligible_track_ids)
    ].copy()

    print("Loading appearance encoder...")
    encoder = AppearanceEncoder(
        args.appearance_model,
        args.device,
    )
    embeddings, sample_counts = extract_track_embeddings(
        video,
        eligible,
        encoder,
        args.samples_per_track,
    )

    capture = cv2.VideoCapture(str(video))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()

    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    capture = cv2.VideoCapture(str(video))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    video_duration_sec = total_frames / fps if total_frames > 0 else 0.0

    print("Estimating internal crop motion...")
    internal_motion = estimate_internal_motion(
        video,
        eligible,
        args.motion_samples_per_track,
    )

    for track_id, quality in quality_by_track.items():
        motion = internal_motion.get(track_id, {})
        quality.update(motion)

        info = track_info[track_id]
        intro_only = (
            info.last_time <= args.intro_outro_window_sec
        )
        outro_only = (
            video_duration_sec > 0
            and info.first_time
            >= max(0.0, video_duration_sec - args.intro_outro_window_sec)
        )
        boundary_only = intro_only or outro_only

        median_internal = quality.get(
            "median_internal_motion",
            float("nan"),
        )
        nearly_pixel_static = (
            np.isfinite(median_internal)
            and median_internal <= args.internal_motion_threshold
        )

        # Conservative image/screen rule: require scene-boundary context and
        # both low box motion and low internal pixel motion.
        likely_image_or_screen = (
            boundary_only
            and quality["static_suspect"]
            and nearly_pixel_static
        )

        if likely_image_or_screen:
            quality["track_classification"] = "likely_intro_outro_image"
        elif quality["static_suspect"] and not nearly_pixel_static:
            quality["track_classification"] = "static_real_candidate"
        elif quality["static_suspect"]:
            quality["track_classification"] = "static_uncertain"
        elif quality["sparse_suspect"]:
            quality["track_classification"] = "sparse_candidate"
        else:
            quality["track_classification"] = "active_real_candidate"

        quality["intro_only"] = intro_only
        quality["outro_only"] = outro_only
        quality["boundary_only"] = boundary_only
        quality["nearly_pixel_static"] = nearly_pixel_static
        quality["likely_image_or_screen"] = likely_image_or_screen

    track_ids = sorted(track_info)

    parent = {track_id: track_id for track_id in track_ids}
    members = {
        track_id: {track_id}
        for track_id in track_ids
    }

    def find(track_id: int) -> int:
        while parent[track_id] != track_id:
            parent[track_id] = parent[parent[track_id]]
            track_id = parent[track_id]
        return track_id

    pair_records: list[dict[str, Any]] = []
    candidate_pairs: list[
        tuple[int, float, float, int, float, str, int, int]
    ] = []

    for index, left in enumerate(track_ids):
        for right in track_ids[index + 1:]:
            common_frames = np.intersect1d(
                observed_frames[left],
                observed_frames[right],
                assume_unique=True,
            )

            common_count = int(len(common_frames))
            overlap_denominator = max(
                1,
                min(
                    len(observed_frames[left]),
                    len(observed_frames[right]),
                ),
            )
            overlap_fraction = common_count / overlap_denominator

            ious = [
                bbox_iou(
                    boxes_by_track[left][int(frame)],
                    boxes_by_track[right][int(frame)],
                )
                for frame in common_frames
            ]
            geometry = [
                box_geometry_metrics(
                    boxes_by_track[left][int(frame)],
                    boxes_by_track[right][int(frame)],
                )
                for frame in common_frames
            ]

            median_normalized_edge_gap = (
                float(np.median(
                    [item["normalized_edge_gap"] for item in geometry]
                ))
                if geometry
                else float("nan")
            )
            median_normalized_center_distance = (
                float(np.median(
                    [item["normalized_center_distance"] for item in geometry]
                ))
                if geometry
                else float("nan")
            )
            median_intersection_over_smaller = (
                float(np.median(
                    [item["intersection_over_smaller"] for item in geometry]
                ))
                if geometry
                else float("nan")
            )
            median_area_ratio = (
                float(np.median(
                    [item["area_ratio"] for item in geometry]
                ))
                if geometry
                else float("nan")
            )

            median_iou = (
                float(np.median(ious))
                if ious
                else float("nan")
            )
            mean_iou = (
                float(np.mean(ious))
                if ious
                else float("nan")
            )

            appearance_similarity = (
                cosine(embeddings[left], embeddings[right])
                if left in embeddings and right in embeddings
                else float("nan")
            )

            frame_gap = nearest_frame_gap(
                observed_frames[left],
                observed_frames[right],
            )
            gap_seconds = frame_gap / fps

            substantial_different_dog_overlap = (
                common_count >= args.substantial_overlap_frames
                and np.isfinite(median_iou)
                and median_iou < args.different_dog_iou_threshold
            )

            duplicate_overlap = (
                common_count >= args.min_common_frames
                and np.isfinite(median_iou)
                and median_iou >= args.duplicate_iou_threshold
                and np.isfinite(appearance_similarity)
                and appearance_similarity
                >= args.duplicate_appearance_threshold
            )

            part_based_duplicate = (
                common_count >= args.min_common_frames
                and np.isfinite(appearance_similarity)
                and appearance_similarity
                >= args.part_duplicate_appearance_threshold
                and (
                    (
                        np.isfinite(median_iou)
                        and median_iou >= args.part_duplicate_min_iou
                    )
                    or (
                        np.isfinite(median_normalized_edge_gap)
                        and median_normalized_edge_gap
                        <= args.part_duplicate_max_normalized_gap
                    )
                )
                # Keep this conservative: the boxes should be spatially related,
                # not merely two visually similar dogs across the frame.
                and np.isfinite(median_normalized_center_distance)
                and median_normalized_center_distance <= 1.50
            )

            fragmented_reid = (
                not substantial_different_dog_overlap
                and common_count < args.substantial_overlap_frames
                and gap_seconds <= args.max_reid_gap_sec
                and np.isfinite(appearance_similarity)
                and appearance_similarity
                >= args.visual_similarity_threshold
            )

            if duplicate_overlap:
                decision = "candidate_duplicate_overlap"
                priority = 2
                candidate_pairs.append(
                    (
                        priority,
                        median_iou,
                        appearance_similarity,
                        -frame_gap,
                        gap_seconds,
                        decision,
                        left,
                        right,
                    )
                )
            elif part_based_duplicate:
                decision = "candidate_part_based_duplicate"
                priority = 3
                candidate_pairs.append(
                    (
                        priority,
                        appearance_similarity,
                        -median_normalized_edge_gap
                        if np.isfinite(median_normalized_edge_gap)
                        else -999.0,
                        -frame_gap,
                        gap_seconds,
                        decision,
                        left,
                        right,
                    )
                )
            elif fragmented_reid:
                decision = "candidate_fragment_reid"
                priority = 1
                candidate_pairs.append(
                    (
                        priority,
                        appearance_similarity,
                        median_iou if np.isfinite(median_iou) else -1.0,
                        -frame_gap,
                        gap_seconds,
                        decision,
                        left,
                        right,
                    )
                )
            elif substantial_different_dog_overlap:
                decision = "blocked_simultaneous_different_dogs"
            else:
                decision = "below_merge_threshold"

            pair_records.append(
                {
                    "left_track_id": left,
                    "right_track_id": right,
                    "common_observed_frames": common_count,
                    "overlap_fraction_of_shorter_track": overlap_fraction,
                    "median_bbox_iou": median_iou,
                    "mean_bbox_iou": mean_iou,
                    "median_normalized_edge_gap": median_normalized_edge_gap,
                    "median_normalized_center_distance": (
                        median_normalized_center_distance
                    ),
                    "median_intersection_over_smaller": (
                        median_intersection_over_smaller
                    ),
                    "median_area_ratio": median_area_ratio,
                    "appearance_similarity": appearance_similarity,
                    "nearest_frame_gap": frame_gap,
                    "nearest_time_gap_sec": gap_seconds,
                    "pair_decision": decision,
                }
            )

    def clusters_compatible(
        root_left: int,
        root_right: int,
    ) -> bool:
        """Reject only clear simultaneous different-dog evidence."""
        for left in members[root_left]:
            for right in members[root_right]:
                common_frames = np.intersect1d(
                    observed_frames[left],
                    observed_frames[right],
                    assume_unique=True,
                )

                if len(common_frames) < args.substantial_overlap_frames:
                    continue

                ious = [
                    bbox_iou(
                        boxes_by_track[left][int(frame)],
                        boxes_by_track[right][int(frame)],
                    )
                    for frame in common_frames
                ]
                median_iou = float(np.median(ious)) if ious else 0.0

                if median_iou < args.different_dog_iou_threshold:
                    return False

        return True

    merge_records: list[dict[str, Any]] = []

    for (
        _priority,
        primary_score,
        secondary_score,
        _negative_gap,
        gap_seconds,
        decision,
        left,
        right,
    ) in sorted(candidate_pairs, reverse=True):
        root_left = find(left)
        root_right = find(right)

        if root_left == root_right:
            continue

        # Simultaneous body-part duplicates are expected to coexist. For this
        # specific high-confidence geometric case, do not reject the merge just
        # because the tracks are visible at the same time.
        if (
            decision != "candidate_part_based_duplicate"
            and not clusters_compatible(root_left, root_right)
        ):
            continue

        if len(members[root_left]) < len(members[root_right]):
            root_left, root_right = root_right, root_left

        parent[root_right] = root_left
        members[root_left].update(members.pop(root_right))

        pair = next(
            row
            for row in pair_records
            if row["left_track_id"] == min(left, right)
            and row["right_track_id"] == max(left, right)
        )

        merge_records.append(
            {
                "left_track_id": left,
                "right_track_id": right,
                "merge_reason": decision,
                "appearance_similarity": pair[
                    "appearance_similarity"
                ],
                "common_observed_frames": pair[
                    "common_observed_frames"
                ],
                "median_bbox_iou": pair["median_bbox_iou"],
                "median_normalized_edge_gap": pair.get(
                    "median_normalized_edge_gap"
                ),
                "median_normalized_center_distance": pair.get(
                    "median_normalized_center_distance"
                ),
                "median_intersection_over_smaller": pair.get(
                    "median_intersection_over_smaller"
                ),
                "time_gap_sec": gap_seconds,
            }
        )

    clusters: dict[int, list[int]] = {}
    for track_id in track_ids:
        clusters.setdefault(find(track_id), []).append(track_id)

    # v2.3 no longer suppresses a track merely because it is static. Real dogs
    # may lie still. Sparse standalone fragments can still be suppressed, and
    # likely intro/outro images are suppressed only when explicitly requested.
    filtered_clusters: dict[int, list[int]] = {}
    suppressed_rows: list[dict[str, Any]] = []
    uncertain_rows: list[dict[str, Any]] = []

    for root_id, ids in clusters.items():
        has_nonsparse_member = any(
            not quality_by_track[track_id]["sparse_suspect"]
            for track_id in ids
        )
        all_likely_images = all(
            quality_by_track[track_id]["likely_image_or_screen"]
            for track_id in ids
        )

        suppress_reason = None
        if not has_nonsparse_member:
            suppress_reason = "standalone_sparse_track"
        elif args.suppress_likely_images and all_likely_images:
            suppress_reason = "likely_intro_outro_image"

        if suppress_reason is not None:
            for track_id in ids:
                suppressed_rows.append(
                    {
                        "track_id": track_id,
                        "reason": suppress_reason,
                        **quality_by_track[track_id],
                    }
                )
            continue

        filtered_clusters[root_id] = ids

        for track_id in ids:
            classification = quality_by_track[track_id][
                "track_classification"
            ]
            if classification in {
                "likely_intro_outro_image",
                "static_uncertain",
            }:
                uncertain_rows.append(
                    {
                        "track_id": track_id,
                        "classification": classification,
                        **quality_by_track[track_id],
                    }
                )

    clusters = filtered_clusters

    ordered_clusters = sorted(
        clusters.values(),
        key=lambda ids: min(
            track_info[track_id].first_time
            for track_id in ids
        ),
    )

    mapping_rows: list[dict[str, Any]] = []
    profiles: dict[str, Any] = {}

    npz_payload = {
        f"track_{track_id}": embedding
        for track_id, embedding in embeddings.items()
    }

    for index, ids in enumerate(ordered_clusters, start=1):
        profile_id = f"dog_{index}"

        profile_embeddings = [
            embeddings[track_id]
            for track_id in ids
            if track_id in embeddings
        ]

        visual_centroid = None
        if profile_embeddings:
            centroid = np.mean(
                np.stack(profile_embeddings),
                axis=0,
            )
            centroid /= np.linalg.norm(centroid) + 1e-12
            visual_centroid = centroid.astype(np.float32)
            npz_payload[f"profile_{profile_id}"] = visual_centroid

        profile_classifications = sorted(
            {
                quality_by_track[track_id]["track_classification"]
                for track_id in ids
            }
        )
        profiles[profile_id] = {
            "track_ids": sorted(ids),
            "visual_track_classifications": profile_classifications,
            "contains_uncertain_visual_track": any(
                classification in {
                    "likely_intro_outro_image",
                    "static_uncertain",
                }
                for classification in profile_classifications
            ),
            "first_seen_time": min(
                track_info[track_id].first_time
                for track_id in ids
            ),
            "last_seen_time": max(
                track_info[track_id].last_time
                for track_id in ids
            ),
            "track_fragments": len(ids),
            "visual_embedding_available": (
                visual_centroid is not None
            ),
            "bark_events": [],
            "bark_count": 0,
            "status": "silent",
        }

        if visual_centroid is not None:
            profiles[profile_id][
                "visual_centroid_key"
            ] = profile_id

        for track_id in sorted(ids):
            info = track_info[track_id]
            mapping_rows.append(
                {
                    "track_id": track_id,
                    "persistent_profile_id": profile_id,
                    "first_frame": info.first_frame,
                    "last_frame": info.last_frame,
                    "first_time_sec": info.first_time,
                    "last_time_sec": info.last_time,
                    "track_rows": info.rows,
                    "mean_confidence": info.mean_confidence,
                    "embedding_samples": sample_counts.get(
                        track_id,
                        0,
                    ),
                    "track_classification": quality_by_track[
                        track_id
                    ]["track_classification"],
                    "median_internal_motion": quality_by_track[
                        track_id
                    ].get("median_internal_motion", float("nan")),
                    "intro_only": quality_by_track[
                        track_id
                    ]["intro_only"],
                    "outro_only": quality_by_track[
                        track_id
                    ]["outro_only"],
                }
            )

    np.savez_compressed(
        output_dir / "track_visual_embeddings.npz",
        **npz_payload,
    )

    pd.DataFrame(mapping_rows).to_csv(
        mapping_path,
        index=False,
    )

    merge_columns = [
        "left_track_id",
        "right_track_id",
        "merge_reason",
        "appearance_similarity",
        "common_observed_frames",
        "median_bbox_iou",
        "median_normalized_edge_gap",
        "median_normalized_center_distance",
        "median_intersection_over_smaller",
        "time_gap_sec",
    ]
    pd.DataFrame(
        merge_records,
        columns=merge_columns,
    ).to_csv(
        output_dir / "visual_track_merges.csv",
        index=False,
    )

    pd.DataFrame(pair_records).to_csv(
        output_dir / "visual_track_pair_diagnostics.csv",
        index=False,
    )

    quality_rows = [
        {"track_id": track_id, **quality}
        for track_id, quality in sorted(quality_by_track.items())
    ]
    pd.DataFrame(quality_rows).to_csv(
        output_dir / "visual_track_quality.csv",
        index=False,
    )
    pd.DataFrame(suppressed_rows).to_csv(
        output_dir / "suppressed_visual_tracks.csv",
        index=False,
    )
    pd.DataFrame(uncertain_rows).to_csv(
        output_dir / "uncertain_visual_tracks.csv",
        index=False,
    )

    summary = {
        "version": "v2.4",
        "temporary_track_ids": len(track_ids),
        "suppressed_track_ids": len(suppressed_rows),
        "uncertain_track_ids": len(uncertain_rows),
        "intro_outro_window_sec": args.intro_outro_window_sec,
        "internal_motion_threshold": args.internal_motion_threshold,
        "suppress_likely_images": args.suppress_likely_images,
        "persistent_all_dog_profiles": len(profiles),
        "track_fragments_merged": len(track_ids) - len(profiles),
        "visual_similarity_threshold": (
            args.visual_similarity_threshold
        ),
        "duplicate_iou_threshold": (
            args.duplicate_iou_threshold
        ),
        "duplicate_appearance_threshold": (
            args.duplicate_appearance_threshold
        ),
        "part_duplicate_appearance_threshold": (
            args.part_duplicate_appearance_threshold
        ),
        "part_duplicate_max_normalized_gap": (
            args.part_duplicate_max_normalized_gap
        ),
        "part_duplicate_min_iou": args.part_duplicate_min_iou,
        "different_dog_iou_threshold": (
            args.different_dog_iou_threshold
        ),
        "max_reid_gap_sec": args.max_reid_gap_sec,
        "profiles": profiles,
    }

    (
        output_dir / "all_dog_visual_profiles.json"
    ).write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("Temporary tracks:", len(track_ids))
    print("Persistent all-dog profiles:", len(profiles))
    print(
        "Fragments merged:",
        len(track_ids) - len(profiles),
    )
    print("Mapping:", mapping_path)
    print(
        "Pair diagnostics:",
        output_dir / "visual_track_pair_diagnostics.csv",
    )


if __name__ == "__main__":
    main()
