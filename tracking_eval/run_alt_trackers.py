#!/usr/bin/env python3
"""
run_alt_trackers.py

Runs SORT, ByteTrack, and a real BoT-SORT (RT-DETR detections + OSNet ReID
embeddings pulled from actual video crops) directly against the original
videos in --videos-dir. Self-contained: the only other files it touches are
track_dogs.py (for the detector/embedding loading) and run_mot_eval.py (for
the shared MOT-txt writer), both imported, not re-implemented.

Expected layout:
    MOT_eval/gt/<seq>.txt        # ground truth - defines which sequences to run
    <videos-dir>/<seq>.<ext>     # original video per sequence
                                  # (ext auto-detected: mp4/mov/avi/mkv/m4v)

Output:
    MOT_eval/trackers/sort_tracker/<seq>.txt
    MOT_eval/trackers/bytetrack_tracker/<seq>.txt
    MOT_eval/trackers/botsort_tracker/<seq>.txt

This needs a GPU plus the same detector/appearance weights as track_dogs.py,
and it re-runs detection from scratch per video, so it's slow. Use
--tracking-fps to subsample if that's too slow.

Then evaluate everything together:
    python run_mot_eval.py --root MOT_eval --only-sampled-frames

Usage:
    python run_alt_trackers.py --videos-dir dataset --root MOT_eval
    python run_alt_trackers.py --videos-dir dataset --root MOT_eval \
        --methods botsort --tracking-fps 10
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

# track_dogs.py (detector/embedding loading) and run_mot_eval.py (shared
# MOT-txt writer) are separate files this script depends on - both must sit
# in the same directory as this script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from track_dogs import AppearanceExtractor, read_video_metadata  # noqa: E402
from run_mot_eval import write_mot_rows  # noqa: E402

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".m4v")


# --------------------------------------------------------------------------
# Shared geometry helpers (previously imported - now defined directly here).
# --------------------------------------------------------------------------
def xywh_to_xyxy(box):
    x, y, w, h = box
    return np.array([x, y, x + w, y + h], dtype=np.float64)


def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return (x1, y1, x2 - x1, y2 - y1)


def iou_batch(boxes_a, boxes_b):
    """boxes_a: (N,4) xyxy, boxes_b: (M,4) xyxy -> (N,M) IoU matrix."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))
    boxes_a = np.expand_dims(boxes_a, 1)  # (N,1,4)
    boxes_b = np.expand_dims(boxes_b, 0)  # (1,M,4)

    xx1 = np.maximum(boxes_a[..., 0], boxes_b[..., 0])
    yy1 = np.maximum(boxes_a[..., 1], boxes_b[..., 1])
    xx2 = np.minimum(boxes_a[..., 2], boxes_b[..., 2])
    yy2 = np.minimum(boxes_a[..., 3], boxes_b[..., 3])

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h

    area_a = np.maximum(0.0, boxes_a[..., 2] - boxes_a[..., 0]) * np.maximum(0.0, boxes_a[..., 3] - boxes_a[..., 1])
    area_b = np.maximum(0.0, boxes_b[..., 2] - boxes_b[..., 0]) * np.maximum(0.0, boxes_b[..., 3] - boxes_b[..., 1])
    union = area_a + area_b - inter

    return np.where(union > 0, inter / union, 0.0)


def linear_assignment_on_iou(iou_matrix, iou_threshold):
    """Hungarian match maximizing IoU, then drop pairs below threshold.
    Returns (matches, unmatched_row_idx, unmatched_col_idx)."""
    if iou_matrix.size == 0:
        return [], list(range(iou_matrix.shape[0])), list(range(iou_matrix.shape[1]))

    row_idx, col_idx = linear_sum_assignment(-iou_matrix)
    matches, matched_rows, matched_cols = [], set(), set()
    for r, c in zip(row_idx, col_idx):
        if iou_matrix[r, c] >= iou_threshold:
            matches.append((r, c))
            matched_rows.add(r)
            matched_cols.add(c)

    unmatched_rows = [r for r in range(iou_matrix.shape[0]) if r not in matched_rows]
    unmatched_cols = [c for c in range(iou_matrix.shape[1]) if c not in matched_cols]
    return matches, unmatched_rows, unmatched_cols


class KalmanBoxTracker:
    """Constant-velocity Kalman tracker on [cx, cy, area, aspect_ratio],
    the standard formulation used by SORT-style trackers (no appearance)."""

    _next_id = 1

    def __init__(self, xyxy_box):
        self.kf = cv2.KalmanFilter(7, 4)
        self.kf.transitionMatrix = np.eye(7, dtype=np.float32)
        for i in range(3):
            self.kf.transitionMatrix[i, i + 4] = 1.0
        self.kf.measurementMatrix = np.eye(4, 7, dtype=np.float32)

        self.kf.processNoiseCov = np.eye(7, dtype=np.float32)
        self.kf.processNoiseCov[4:, 4:] *= 0.01
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32)
        self.kf.measurementNoiseCov[2:, 2:] *= 10.0
        self.kf.errorCovPost = np.eye(7, dtype=np.float32) * 10.0

        z = self._box_to_z(xyxy_box)
        self.kf.statePost = np.zeros((7, 1), dtype=np.float32)
        self.kf.statePost[:4, 0] = z
        self.kf.statePre = self.kf.statePost.copy()

        self.id = KalmanBoxTracker._next_id
        KalmanBoxTracker._next_id += 1

        self.time_since_update = 0
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.last_conf = 1.0

    @staticmethod
    def _box_to_z(xyxy_box):
        x1, y1, x2, y2 = xyxy_box
        w = max(1e-3, x2 - x1)
        h = max(1e-3, y2 - y1)
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0
        area = w * h
        aspect = w / h
        return np.array([cx, cy, area, aspect], dtype=np.float32)

    @staticmethod
    def _z_to_box(z):
        cx, cy, area, aspect = z
        area = max(1e-3, area)
        aspect = max(1e-3, aspect)
        w = np.sqrt(area * aspect)
        h = area / w if w > 0 else 1e-3
        return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dtype=np.float64)

    def predict(self):
        pred = self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self._z_to_box(pred[:4, 0])

    def update(self, xyxy_box, conf):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.last_conf = conf
        z = self._box_to_z(xyxy_box).reshape(4, 1)
        self.kf.correct(z)

    def current_box(self):
        return self._z_to_box(self.kf.statePost[:4, 0])


def find_video(videos_dir: Path, seq: str) -> Path | None:
    for ext in VIDEO_EXTENSIONS:
        candidate = videos_dir / f"{seq}{ext}"
        if candidate.is_file():
            return candidate
    return None


def discover_sequences(root: Path) -> list[str]:
    gt_dir = root / "gt"
    return sorted(
        (p.stem for p in gt_dir.glob("*.txt") if p.stat().st_size > 0),
        key=lambda s: (len(s), s),
    )


# --------------------------------------------------------------------------
# ReID-aware Kalman tracker: same constant-velocity motion model as
# run_alt_trackers.KalmanBoxTracker, plus an EMA-smoothed appearance
# embedding (mirrors track_dogs.StableTracker.update_embedding).
# --------------------------------------------------------------------------
class ReIDKalmanBoxTracker:
    _next_id = 1

    def __init__(self, xyxy_box, embedding):
        self.kf = cv2.KalmanFilter(7, 4)
        self.kf.transitionMatrix = np.eye(7, dtype=np.float32)
        for i in range(3):
            self.kf.transitionMatrix[i, i + 4] = 1.0
        self.kf.measurementMatrix = np.eye(4, 7, dtype=np.float32)
        self.kf.processNoiseCov = np.eye(7, dtype=np.float32)
        self.kf.processNoiseCov[4:, 4:] *= 0.01
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32)
        self.kf.measurementNoiseCov[2:, 2:] *= 10.0
        self.kf.errorCovPost = np.eye(7, dtype=np.float32) * 10.0

        z = self._box_to_z(xyxy_box)
        self.kf.statePost = np.zeros((7, 1), dtype=np.float32)
        self.kf.statePost[:4, 0] = z
        self.kf.statePre = self.kf.statePost.copy()

        self.id = ReIDKalmanBoxTracker._next_id
        ReIDKalmanBoxTracker._next_id += 1

        self.time_since_update = 0
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.last_conf = 1.0
        self.embedding = (
            embedding if embedding is not None and np.any(embedding) else None
        )

    @staticmethod
    def _box_to_z(xyxy_box):
        x1, y1, x2, y2 = xyxy_box
        w = max(1e-3, x2 - x1)
        h = max(1e-3, y2 - y1)
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0
        return np.array([cx, cy, w * h, w / h], dtype=np.float32)

    @staticmethod
    def _z_to_box(z):
        cx, cy, area, aspect = z
        area = max(1e-3, area)
        aspect = max(1e-3, aspect)
        w = np.sqrt(area * aspect)
        h = area / w if w > 0 else 1e-3
        return np.array(
            [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0],
            dtype=np.float64,
        )

    def predict(self):
        pred = self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self._z_to_box(pred[:4, 0])

    def update(self, xyxy_box, conf, embedding, appearance_ema):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.last_conf = conf
        self.kf.correct(self._box_to_z(xyxy_box).reshape(4, 1))

        if embedding is not None and np.any(embedding):
            if self.embedding is None:
                self.embedding = embedding
            else:
                blended = (
                    appearance_ema * embedding
                    + (1.0 - appearance_ema) * self.embedding
                )
                norm = np.linalg.norm(blended)
                self.embedding = blended / norm if norm > 0 else blended

    def current_box(self):
        return self._z_to_box(self.kf.statePost[:4, 0])


def _appearance_cost_matrix(det_embeddings, trk_embeddings):
    n, m = len(det_embeddings), len(trk_embeddings)
    cost = np.full((n, m), np.nan, dtype=np.float64)
    for i, de in enumerate(det_embeddings):
        if de is None or not np.any(de):
            continue
        for j, te in enumerate(trk_embeddings):
            if te is None or not np.any(te):
                continue
            cost[i, j] = 1.0 - float(np.dot(de, te))
    return cost


def _combined_cost_matrix(iou_mat, app_cost, iou_weight):
    cost = 1.0 - iou_mat
    have_app = ~np.isnan(app_cost)
    return np.where(
        have_app,
        iou_weight * (1.0 - iou_mat) + (1.0 - iou_weight) * app_cost,
        cost,
    )


def _hungarian_on_cost(cost_matrix, max_cost=1e5):
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    row_idx, col_idx = linear_sum_assignment(cost_matrix)
    matches, matched_rows, matched_cols = [], set(), set()
    for r, c in zip(row_idx, col_idx):
        if cost_matrix[r, c] < max_cost:
            matches.append((r, c))
            matched_rows.add(r)
            matched_cols.add(c)
    unmatched_rows = [r for r in range(cost_matrix.shape[0]) if r not in matched_rows]
    unmatched_cols = [c for c in range(cost_matrix.shape[1]) if c not in matched_cols]
    return matches, unmatched_rows, unmatched_cols


def run_botsort(
    frames,
    iou_threshold=0.3,
    max_age=25,
    min_hits=1,
    new_track_thresh=0.6,
    track_low_thresh=0.1,
    appearance_ema=0.30,
    iou_weight=0.60,
    appearance_gate=0.70,
):
    """frames: sorted list of (frame_i, [(x,y,w,h,conf), ...], [embedding|None, ...])."""
    ReIDKalmanBoxTracker._next_id = 1
    trackers: list[ReIDKalmanBoxTracker] = []
    output_rows = []
    first_frame = frames[0][0] if frames else 0

    for frame_i, dets, embeddings in frames:
        high_idx = [i for i, d in enumerate(dets) if d[4] >= new_track_thresh]
        low_idx = [i for i, d in enumerate(dets) if track_low_thresh <= d[4] < new_track_thresh]

        pred_boxes = np.array([t.predict() for t in trackers]) if trackers else np.empty((0, 4))
        trk_embeddings = [t.embedding for t in trackers]

        # Stage 1: high-confidence detections vs. all trackers, IoU + appearance.
        high_boxes = (
            np.array([xywh_to_xyxy(dets[i][:4]) for i in high_idx]) if high_idx else np.empty((0, 4))
        )
        high_emb = [embeddings[i] for i in high_idx]

        iou1 = iou_batch(high_boxes, pred_boxes)
        app1 = _appearance_cost_matrix(high_emb, trk_embeddings)
        cost1 = _combined_cost_matrix(iou1, app1, iou_weight)

        # Gate: reject a pair unless IoU clears the threshold OR appearance
        # similarity clears the gate (lets ReID rescue a match through a
        # brief occlusion where boxes barely overlap).
        have_app1 = ~np.isnan(app1)
        appearance_ok1 = have_app1 & (app1 <= (1.0 - appearance_gate))
        reject1 = (iou1 < iou_threshold) & ~appearance_ok1
        cost1 = np.where(reject1, 1e6, cost1)

        matches1, unmatched_high_local, unmatched_trk1 = _hungarian_on_cost(cost1)

        for det_local, trk_idx in matches1:
            det_i = high_idx[det_local]
            trackers[trk_idx].update(
                high_boxes[det_local], dets[det_i][4], high_emb[det_local], appearance_ema
            )

        # Stage 2: leftover trackers vs. low-confidence detections, IoU only
        # (mirrors ByteTrack's second stage - no ReID gate here on purpose,
        # since low-confidence boxes are noisier).
        remaining_trk_idx = unmatched_trk1
        low_boxes = (
            np.array([xywh_to_xyxy(dets[i][:4]) for i in low_idx]) if low_idx else np.empty((0, 4))
        )
        low_emb = [embeddings[i] for i in low_idx]
        remaining_pred_boxes = pred_boxes[remaining_trk_idx] if remaining_trk_idx else np.empty((0, 4))

        iou2 = iou_batch(low_boxes, remaining_pred_boxes)
        matches2, _unmatched_low, _unmatched_trk2 = linear_assignment_on_iou(iou2, iou_threshold)

        for det_local, local_trk in matches2:
            trk_idx = remaining_trk_idx[local_trk]
            det_i = low_idx[det_local]
            trackers[trk_idx].update(
                low_boxes[det_local], dets[det_i][4], low_emb[det_local], appearance_ema
            )

        # New tracks only from high-confidence, still-unmatched detections.
        for det_local in unmatched_high_local:
            det_i = high_idx[det_local]
            trackers.append(ReIDKalmanBoxTracker(high_boxes[det_local], high_emb[det_local]))
            trackers[-1].update(
                high_boxes[det_local], dets[det_i][4], high_emb[det_local], appearance_ema
            )

        trackers = [t for t in trackers if t.time_since_update <= max_age]

        for t in trackers:
            if t.time_since_update == 0 and (t.hit_streak >= min_hits or frame_i <= first_frame + min_hits):
                x1, y1, x2, y2 = t.current_box()
                x, y, w, h = xyxy_to_xywh((x1, y1, x2, y2))
                output_rows.append((frame_i, t.id, x, y, w, h, t.last_conf))

    return output_rows


def run_sort(frames, iou_threshold=0.3, max_age=20, min_hits=3):
    KalmanBoxTracker._next_id = 1
    trackers = []
    output_rows = []
    first_frame = frames[0][0] if frames else 0

    for frame_i, dets, _embeddings in frames:
        det_boxes = np.array([xywh_to_xyxy(d[:4]) for d in dets]) if dets else np.empty((0, 4))
        det_confs = [d[4] for d in dets]
        pred_boxes = np.array([t.predict() for t in trackers]) if trackers else np.empty((0, 4))

        iou_matrix = iou_batch(det_boxes, pred_boxes)
        matches, unmatched_dets, _unmatched_trks = linear_assignment_on_iou(iou_matrix, iou_threshold)

        for det_idx, trk_idx in matches:
            trackers[trk_idx].update(det_boxes[det_idx], det_confs[det_idx])
        for det_idx in unmatched_dets:
            trackers.append(KalmanBoxTracker(det_boxes[det_idx]))
            trackers[-1].update(det_boxes[det_idx], det_confs[det_idx])

        trackers = [t for t in trackers if t.time_since_update <= max_age]

        for t in trackers:
            if t.time_since_update == 0 and (t.hit_streak >= min_hits or frame_i <= first_frame + min_hits):
                x1, y1, x2, y2 = t.current_box()
                x, y, w, h = xyxy_to_xywh((x1, y1, x2, y2))
                output_rows.append((frame_i, t.id, x, y, w, h, t.last_conf))

    return output_rows


def run_bytetrack(
    frames,
    high_thresh=0.6,
    low_thresh=0.1,
    new_track_thresh=0.7,
    iou_threshold=0.3,
    max_age=30,
    min_hits=1,
):
    KalmanBoxTracker._next_id = 1
    trackers = []
    output_rows = []
    first_frame = frames[0][0] if frames else 0

    for frame_i, dets, _embeddings in frames:
        high = [d for d in dets if d[4] >= high_thresh]
        low = [d for d in dets if low_thresh <= d[4] < high_thresh]

        pred_boxes = np.array([t.predict() for t in trackers]) if trackers else np.empty((0, 4))

        high_boxes = np.array([xywh_to_xyxy(d[:4]) for d in high]) if high else np.empty((0, 4))
        high_confs = [d[4] for d in high]
        iou1 = iou_batch(high_boxes, pred_boxes)
        matches1, unmatched_high, unmatched_trk1 = linear_assignment_on_iou(iou1, iou_threshold)

        for det_idx, trk_idx in matches1:
            trackers[trk_idx].update(high_boxes[det_idx], high_confs[det_idx])

        low_boxes = np.array([xywh_to_xyxy(d[:4]) for d in low]) if low else np.empty((0, 4))
        low_confs = [d[4] for d in low]
        remaining_pred = pred_boxes[unmatched_trk1] if unmatched_trk1 else np.empty((0, 4))
        iou2 = iou_batch(low_boxes, remaining_pred)
        matches2, _u_low, _u_trk2 = linear_assignment_on_iou(iou2, iou_threshold)
        for det_idx, local_trk in matches2:
            trk_idx = unmatched_trk1[local_trk]
            trackers[trk_idx].update(low_boxes[det_idx], low_confs[det_idx])

        for det_idx in unmatched_high:
            if high_confs[det_idx] >= new_track_thresh:
                trackers.append(KalmanBoxTracker(high_boxes[det_idx]))
                trackers[-1].update(high_boxes[det_idx], high_confs[det_idx])

        trackers = [t for t in trackers if t.time_since_update <= max_age]

        for t in trackers:
            if t.time_since_update == 0 and (t.hit_streak >= min_hits or frame_i <= first_frame + min_hits):
                x1, y1, x2, y2 = t.current_box()
                x, y, w, h = xyxy_to_xywh((x1, y1, x2, y2))
                output_rows.append((frame_i, t.id, x, y, w, h, t.last_conf))

    return output_rows


METHODS = {"sort": run_sort, "bytetrack": run_bytetrack, "botsort": run_botsort}
OUTPUT_FOLDER_NAMES = {
    "sort": "sort_tracker",
    "bytetrack": "bytetrack_tracker",
    "botsort": "botsort_tracker",
}


def detect_video(detector, extractor, video_path, conf_threshold, tracking_fps, device):
    """Runs raw per-frame detection (no built-in tracker ID) + OSNet
    embeddings, matching how track_dogs.py loads its detector/extractor."""
    info = read_video_metadata(video_path)
    source_fps = float(info["fps"])
    frame_step = 1 if tracking_fps is None else max(1, int(round(source_fps / tracking_fps)))

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames = []
    source_frame = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            if source_frame % frame_step == 0:
                results = detector.predict(frame, conf=conf_threshold, verbose=False, device=device)
                result = results[0]

                boxes_xywh, embed_boxes = [], []
                if result.boxes is not None and len(result.boxes):
                    xyxy = result.boxes.xyxy.detach().cpu().numpy()
                    cls_ids = result.boxes.cls.detach().cpu().numpy()
                    confs = result.boxes.conf.detach().cpu().numpy()
                    for box, cls_id, conf in zip(xyxy, cls_ids, confs):
                        name = (
                            detector.names.get(int(cls_id), str(int(cls_id)))
                            if isinstance(detector.names, dict)
                            else detector.names[int(cls_id)]
                        )
                        if str(name).lower() != "dog" or conf < conf_threshold:
                            continue
                        x1, y1, x2, y2 = map(float, box)
                        boxes_xywh.append((x1, y1, x2 - x1, y2 - y1, float(conf)))
                        embed_boxes.append(np.asarray(box, dtype=np.float32))

                embeddings = list(extractor(frame, embed_boxes)) if embed_boxes else []
                frames.append(
                    (source_frame, boxes_xywh, embeddings if embeddings else [None] * len(boxes_xywh))
                )

            source_frame += 1
    finally:
        capture.release()

    return frames, info


def main():
    parser = argparse.ArgumentParser(
        description="Run SORT / ByteTrack / real (ReID) BoT-SORT directly on the original videos."
    )
    parser.add_argument("--videos-dir", type=Path, required=True, help="Folder with original videos, named <seq>.<ext>")
    parser.add_argument("--root", type=Path, required=True, help="MOT_eval folder (sequences discovered from gt/)")
    parser.add_argument("--detector-weights", type=Path, default=Path("rtdetr-l.pt"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--appearance-model", default="osnet_x0_25")
    parser.add_argument("--conf-threshold", type=float, default=0.60)
    parser.add_argument(
        "--tracking-fps",
        type=float,
        default=None,
        help="If unset, every frame is processed (matches track_dogs.py's default).",
    )
    parser.add_argument("--methods", nargs="*", default=list(METHODS.keys()), choices=list(METHODS.keys()))
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--max-age", type=int, default=25)
    parser.add_argument("--appearance-ema", type=float, default=0.30)
    parser.add_argument("--iou-weight", type=float, default=0.60)
    parser.add_argument(
        "--appearance-gate",
        type=float,
        default=0.70,
        help="Cosine similarity that can rescue a match with weak IoU (botsort only).",
    )
    args = parser.parse_args()

    from ultralytics import RTDETR

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is unavailable.")

    root = args.root.expanduser().resolve()
    videos_dir = args.videos_dir.expanduser().resolve()
    seqs = discover_sequences(root)
    if not seqs:
        sys.exit(f"ERROR: no gt .txt files found in {root / 'gt'}")

    print(f"Found {len(seqs)} sequence(s) in {root / 'gt'}")
    print("Loading RT-DETR...")
    detector = RTDETR(str(args.detector_weights))
    print("Loading OSNet...")
    extractor = AppearanceExtractor(args.appearance_model, args.device)
    if str(args.device).startswith("cuda"):
        torch.backends.cudnn.benchmark = True

    per_seq_frames = {}
    for seq in seqs:
        video_path = find_video(videos_dir, seq)
        if video_path is None:
            print(
                f"WARNING: no video found for sequence '{seq}' in {videos_dir} "
                f"(tried {VIDEO_EXTENSIONS}); skipping"
            )
            continue
        print(f"\nDetecting on {video_path.name} ...")
        started = time.perf_counter()
        frames, info = detect_video(
            detector, extractor, video_path, args.conf_threshold, args.tracking_fps, args.device
        )
        elapsed = time.perf_counter() - started
        n_dets = sum(len(f[1]) for f in frames)
        print(
            f"  {info['width']}x{info['height']} @ {info['fps']:.2f}fps | "
            f"{len(frames)} sampled frames, {n_dets} raw detections, {elapsed:.1f}s"
        )
        per_seq_frames[seq] = frames

    if not per_seq_frames:
        sys.exit("ERROR: no sequences had a matching video; nothing to do.")

    for method in args.methods:
        out_folder = root / "trackers" / OUTPUT_FOLDER_NAMES[method]
        out_folder.mkdir(parents=True, exist_ok=True)
        print(f"\nRunning {method} -> trackers/{OUTPUT_FOLDER_NAMES[method]}/")

        for seq, frames in per_seq_frames.items():
            if method == "botsort":
                output_rows = run_botsort(
                    frames,
                    iou_threshold=args.iou_threshold,
                    max_age=args.max_age,
                    appearance_ema=args.appearance_ema,
                    iou_weight=args.iou_weight,
                    appearance_gate=args.appearance_gate,
                )
            else:
                output_rows = METHODS[method](
                    frames, iou_threshold=args.iou_threshold, max_age=args.max_age
                )

            padded_rows = [(f, tid, x, y, w, h, conf, 1, 1) for f, tid, x, y, w, h, conf in output_rows]
            write_mot_rows(padded_rows, str(out_folder / f"{seq}.txt"), is_gt=False)

        print(f"  wrote {len(per_seq_frames)} sequence file(s)")

    print("\nDone. Now run:")
    print(f"  python run_mot_eval.py --root {root} --only-sampled-frames")


if __name__ == "__main__":
    main()