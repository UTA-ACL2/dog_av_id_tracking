#!/usr/bin/env python3
"""
track_dogs.py

Run the single-video dog detection and stable-tracking stage used by the
active-barker inference pipeline.

Input:
    raw video

Outputs:
    <output-dir>/tracks.csv
    <output-dir>/tracking_metadata.json

Tracking stack:
    RT-DETR-L
    Ultralytics BoT-SORT IDs
    OSNet x0.25 appearance embeddings
    custom stable IDs
    Kalman filter
    Hungarian IoU + appearance matching

The implementation is adapted from create-dataset-tmax.py, but removes all
dataset batching, raw_videos.json, sentences.json, CVAT, and manifest logic.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import torchreid
import torchvision.transforms as T
from scipy.optimize import linear_sum_assignment
from ultralytics import RTDETR


DEFAULT_DETECTOR = Path("rtdetr-l.pt")

TRACK_COLUMNS = [
    "frame",
    "time_sec",
    "track_id",
    "x1",
    "y1",
    "x2",
    "y2",
    "confidence",
    "class_name",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect and stably track every dog in one video."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--detector-weights",
        type=Path,
        default=DEFAULT_DETECTOR,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tracking-fps", type=float, default=10.0)
    parser.add_argument("--conf-threshold", type=float, default=0.60)
    parser.add_argument("--iou-threshold", type=float, default=0.08)
    parser.add_argument("--max-age", type=int, default=20)
    parser.add_argument("--appearance-model", default="osnet_x0_25")
    parser.add_argument("--appearance-threshold", type=float, default=0.70)
    parser.add_argument("--appearance-ema", type=float, default=0.30)
    parser.add_argument("--iou-weight", type=float, default=0.60)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing tracks.csv.",
    )
    return parser.parse_args()


def file_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "absolute_path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def read_video_metadata(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()

    if fps <= 0 or width <= 0 or height <= 0 or frames <= 0:
        raise RuntimeError(
            f"Invalid video metadata: fps={fps}, width={width}, "
            f"height={height}, frames={frames}"
        )

    return {
        "fps": fps,
        "width": width,
        "height": height,
        "frames": frames,
        "duration_seconds": frames / fps,
    }


def box_iou(first: np.ndarray, second: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = map(float, first)
    bx1, by1, bx2, by2 = map(float, second)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def xyxy_to_xywh(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = map(float, box)
    return np.array(
        [
            [(x1 + x2) / 2.0],
            [(y1 + y2) / 2.0],
            [x2 - x1],
            [y2 - y1],
        ],
        dtype=np.float32,
    )


def create_kalman_filter() -> cv2.KalmanFilter:
    kf = cv2.KalmanFilter(6, 4)
    kf.transitionMatrix = np.array(
        [
            [1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 1],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    kf.measurementMatrix = np.eye(4, 6, dtype=np.float32)
    kf.processNoiseCov = np.eye(6, dtype=np.float32) * 0.5
    kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.05
    kf.errorCovPost = np.eye(6, dtype=np.float32)
    return kf


class AppearanceExtractor:
    _transform = T.Compose(
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
    def __call__(
        self,
        frame_bgr: np.ndarray,
        boxes: list[np.ndarray],
    ) -> np.ndarray:
        if not boxes:
            return np.empty((0, 512), dtype=np.float32)

        height, width = frame_bgr.shape[:2]
        tensors: list[torch.Tensor] = []
        valid: list[bool] = []

        for box in boxes:
            x1, y1, x2, y2 = map(float, box)
            ix1 = int(max(0, min(width, math.floor(x1))))
            iy1 = int(max(0, min(height, math.floor(y1))))
            ix2 = int(max(0, min(width, math.ceil(x2))))
            iy2 = int(max(0, min(height, math.ceil(y2))))

            if ix2 <= ix1 or iy2 <= iy1:
                tensors.append(torch.zeros((3, 256, 128), dtype=torch.float32))
                valid.append(False)
                continue

            crop = frame_bgr[iy1:iy2, ix1:ix2]
            if crop.size == 0:
                tensors.append(torch.zeros((3, 256, 128), dtype=torch.float32))
                valid.append(False)
                continue

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensors.append(self._transform(crop_rgb))
            valid.append(True)

        batch = torch.stack(tensors).to(self.device)
        embeddings = self.model(batch)
        norms = torch.linalg.vector_norm(
            embeddings,
            dim=1,
            keepdim=True,
        ).clamp(min=1e-6)
        output = (embeddings / norms).cpu().numpy().astype(np.float32)

        for index, is_valid in enumerate(valid):
            if not is_valid:
                output[index] = 0.0

        return output


class StableTracker:
    def __init__(
        self,
        iou_threshold: float,
        max_age: int,
        appearance_extractor: AppearanceExtractor,
        appearance_threshold: float,
        appearance_ema: float,
        iou_weight: float,
    ) -> None:
        self.next_stable_id = 0
        self.active: dict[int, dict[str, Any]] = {}
        self.lost: dict[int, dict[str, Any]] = {}
        self.yolo_to_stable: dict[int, int] = {}

        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.extractor = appearance_extractor
        self.appearance_threshold = appearance_threshold
        self.appearance_ema = appearance_ema
        self.iou_weight = iou_weight

    @staticmethod
    def cosine(first: np.ndarray, second: np.ndarray) -> float:
        return float(np.dot(first, second))

    def update_embedding(
        self,
        stable_id: int,
        new_embedding: np.ndarray,
        store: dict[int, dict[str, Any]],
    ) -> None:
        old_embedding = store[stable_id].get("embedding")

        if old_embedding is None or not np.any(old_embedding):
            store[stable_id]["embedding"] = new_embedding
            return

        blended = (
            self.appearance_ema * new_embedding
            + (1.0 - self.appearance_ema) * old_embedding
        )
        norm = np.linalg.norm(blended)
        store[stable_id]["embedding"] = (
            blended / norm if norm > 0 else blended
        )

    def update(
        self,
        detections: list[tuple[np.ndarray, int]],
        frame_id: int,
        frame_bgr: np.ndarray,
    ) -> list[tuple[np.ndarray, int]]:
        if detections:
            detection_embeddings = self.extractor(
                frame_bgr,
                [box for box, _ in detections],
            )
        else:
            detection_embeddings = np.empty((0, 512), dtype=np.float32)

        updated_this_frame: set[int] = set()
        matched_stable_ids: set[int] = set()
        unmatched_detections: list[tuple[np.ndarray, int]] = []
        unmatched_embeddings: list[np.ndarray | None] = []

        # Stage 1: retain BoT-SORT IDs when geometry remains plausible.
        for index, (box, yolo_id) in enumerate(detections):
            embedding = (
                detection_embeddings[index]
                if len(detection_embeddings)
                else None
            )
            stable_id = self.yolo_to_stable.get(yolo_id)

            if stable_id is None:
                unmatched_detections.append((box, yolo_id))
                unmatched_embeddings.append(embedding)
                continue

            existing = self.active.get(stable_id)
            if existing is not None and box_iou(box, existing["box"]) < 0.5:
                self.yolo_to_stable.pop(yolo_id, None)
                unmatched_detections.append((box, yolo_id))
                unmatched_embeddings.append(embedding)
                continue

            if stable_id in self.lost:
                self.active[stable_id] = self.lost.pop(stable_id)

            if stable_id in self.active:
                self.active[stable_id]["kf"].correct(xyxy_to_xywh(box))
                self.active[stable_id]["box"] = np.asarray(
                    box, dtype=np.float32
                )
                self.active[stable_id]["last_seen"] = frame_id
                self.active[stable_id]["yolo_id"] = yolo_id

                if embedding is not None and np.any(embedding):
                    self.update_embedding(stable_id, embedding, self.active)

                matched_stable_ids.add(stable_id)
                updated_this_frame.add(stable_id)

        # Stage 2: Hungarian IoU + appearance matching.
        if unmatched_detections:
            predicted_lost_boxes: dict[int, np.ndarray] = {}

            for stable_id, data in self.lost.items():
                prediction = data["kf"].predict()
                center_x, center_y, width, height = prediction[:4].flatten()
                predicted_lost_boxes[stable_id] = np.array(
                    [
                        center_x - width / 2.0,
                        center_y - height / 2.0,
                        center_x + width / 2.0,
                        center_y + height / 2.0,
                    ],
                    dtype=np.float32,
                )

            candidate_ids = [
                stable_id
                for stable_id in self.active
                if stable_id not in matched_stable_ids
            ] + list(self.lost.keys())

            candidate_store = {**self.active, **self.lost}
            candidate_boxes = [
                (
                    predicted_lost_boxes[stable_id]
                    if stable_id in predicted_lost_boxes
                    else np.asarray(
                        self.active[stable_id]["box"],
                        dtype=np.float32,
                    )
                )
                for stable_id in candidate_ids
            ]

            if candidate_ids:
                cost_matrix = np.ones(
                    (len(unmatched_detections), len(candidate_ids)),
                    dtype=np.float32,
                )

                for detection_index, ((box, _), embedding) in enumerate(
                    zip(unmatched_detections, unmatched_embeddings)
                ):
                    for candidate_index, (
                        stable_id,
                        candidate_box,
                    ) in enumerate(zip(candidate_ids, candidate_boxes)):
                        iou_cost = 1.0 - box_iou(box, candidate_box)
                        candidate_embedding = candidate_store[
                            stable_id
                        ].get("embedding")

                        if (
                            embedding is not None
                            and candidate_embedding is not None
                            and np.any(embedding)
                            and np.any(candidate_embedding)
                        ):
                            appearance_cost = 1.0 - self.cosine(
                                embedding,
                                candidate_embedding,
                            )
                            cost_matrix[
                                detection_index, candidate_index
                            ] = (
                                self.iou_weight * iou_cost
                                + (1.0 - self.iou_weight) * appearance_cost
                            )
                        else:
                            cost_matrix[
                                detection_index, candidate_index
                            ] = iou_cost

                detection_indices, candidate_indices = linear_sum_assignment(
                    cost_matrix
                )
                assignment = dict(
                    zip(
                        detection_indices.tolist(),
                        candidate_indices.tolist(),
                    )
                )

                remaining_detections: list[tuple[np.ndarray, int]] = []
                remaining_embeddings: list[np.ndarray | None] = []

                for detection_index, ((box, yolo_id), embedding) in enumerate(
                    zip(unmatched_detections, unmatched_embeddings)
                ):
                    if detection_index not in assignment:
                        remaining_detections.append((box, yolo_id))
                        remaining_embeddings.append(embedding)
                        continue

                    candidate_index = assignment[detection_index]
                    stable_id = candidate_ids[candidate_index]
                    candidate_box = candidate_boxes[candidate_index]
                    iou_score = box_iou(box, candidate_box)
                    candidate_embedding = candidate_store[
                        stable_id
                    ].get("embedding")

                    appearance_score = None
                    if (
                        embedding is not None
                        and candidate_embedding is not None
                        and np.any(embedding)
                        and np.any(candidate_embedding)
                    ):
                        appearance_score = self.cosine(
                            embedding,
                            candidate_embedding,
                        )

                    accepted = (
                        iou_score >= self.iou_threshold
                        or (
                            appearance_score is not None
                            and appearance_score
                            >= self.appearance_threshold
                        )
                    )

                    if not accepted:
                        remaining_detections.append((box, yolo_id))
                        remaining_embeddings.append(embedding)
                        continue

                    if stable_id in self.lost:
                        self.active[stable_id] = self.lost.pop(stable_id)

                    previous_yolo_id = self.active[stable_id].get("yolo_id")
                    if previous_yolo_id is not None:
                        self.yolo_to_stable.pop(
                            int(previous_yolo_id),
                            None,
                        )

                    self.active[stable_id]["kf"].correct(xyxy_to_xywh(box))
                    self.active[stable_id]["box"] = np.asarray(
                        box, dtype=np.float32
                    )
                    self.active[stable_id]["last_seen"] = frame_id
                    self.active[stable_id]["yolo_id"] = yolo_id
                    self.yolo_to_stable[yolo_id] = stable_id

                    if embedding is not None and np.any(embedding):
                        self.update_embedding(
                            stable_id,
                            embedding,
                            self.active,
                        )

                    matched_stable_ids.add(stable_id)
                    updated_this_frame.add(stable_id)

                unmatched_detections = remaining_detections
                unmatched_embeddings = remaining_embeddings

        # Stage 3: unmatched detections become new stable IDs.
        for (box, yolo_id), embedding in zip(
            unmatched_detections,
            unmatched_embeddings,
        ):
            stable_id = self.next_stable_id
            self.next_stable_id += 1

            kf = create_kalman_filter()
            center_x, center_y, width, height = xyxy_to_xywh(box).flatten()
            state = np.array(
                [
                    [center_x],
                    [center_y],
                    [width],
                    [height],
                    [0],
                    [0],
                ],
                dtype=np.float32,
            )
            kf.statePre = state.copy()
            kf.statePost = state.copy()

            self.yolo_to_stable[yolo_id] = stable_id
            self.active[stable_id] = {
                "kf": kf,
                "box": np.asarray(box, dtype=np.float32),
                "last_seen": frame_id,
                "yolo_id": yolo_id,
                "embedding": embedding,
            }
            updated_this_frame.add(stable_id)

        for stable_id, data in list(self.active.items()):
            if frame_id - int(data["last_seen"]) > self.max_age:
                self.lost[stable_id] = self.active.pop(stable_id)

        return [
            (
                np.asarray(
                    self.active[stable_id]["box"],
                    dtype=np.float32,
                ),
                stable_id,
            )
            for stable_id in updated_this_frame
            if stable_id in self.active
        ]


def main() -> None:
    args = parse_args()

    video = args.video.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    tracks_path = output_dir / "tracks.csv"
    metadata_path = output_dir / "tracking_metadata.json"

    if not video.is_file():
        raise FileNotFoundError(video)

    if tracks_path.exists() and not args.force:
        raise FileExistsError(
            f"{tracks_path} already exists. Use --force to replace it."
        )

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is unavailable.")

    # A bare model name such as rtdetr-l.pt may be downloaded by Ultralytics.
    # Explicit paths must already exist.
    if args.detector_weights.parent != Path("."):
        if not args.detector_weights.is_file():
            raise FileNotFoundError(args.detector_weights)

    if args.tracking_fps <= 0:
        raise ValueError("--tracking-fps must be positive.")

    output_dir.mkdir(parents=True, exist_ok=True)
    info = read_video_metadata(video)

    source_fps = float(info["fps"])
    frame_step = max(
        1,
        int(round(source_fps / args.tracking_fps)),
    )
    effective_tracking_fps = source_fps / frame_step

    print("=" * 78)
    print("DOG TRACKING")
    print("=" * 78)
    print("Video:", video)
    print(
        f"Metadata: {info['width']}x{info['height']}, "
        f"{source_fps:.3f} FPS, {info['frames']} frames"
    )
    print("Tracking frame step:", frame_step)
    print("Effective tracking FPS:", effective_tracking_fps)
    print("Detector:", args.detector_weights)
    print("Device:", args.device)

    started = time.perf_counter()

    print("Loading RT-DETR...")
    detector = RTDETR(str(args.detector_weights))

    print("Loading OSNet...")
    appearance_extractor = AppearanceExtractor(
        args.appearance_model,
        args.device,
    )

    if str(args.device).startswith("cuda"):
        torch.backends.cudnn.benchmark = True

    stable_tracker = StableTracker(
        iou_threshold=args.iou_threshold,
        max_age=args.max_age,
        appearance_extractor=appearance_extractor,
        appearance_threshold=args.appearance_threshold,
        appearance_ema=args.appearance_ema,
        iou_weight=args.iou_weight,
    )

    # Prevent tracker state from a previous invocation from leaking in.
    try:
        detector.predictor = None
    except Exception:
        pass

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")

    rows: list[dict[str, Any]] = []
    sampled_frames = 0
    source_frame = 0

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            if source_frame % frame_step == 0:
                sampled_frames += 1

                results = detector.track(
                    frame,
                    persist=True,
                    tracker="botsort.yaml",
                    conf=args.conf_threshold,
                    verbose=False,
                    device=args.device,
                )
                result = results[0]

                detections: list[tuple[np.ndarray, int]] = []
                confidence_lookup: dict[int, float] = {}

                if (
                    result.boxes is not None
                    and result.boxes.id is not None
                ):
                    boxes = result.boxes.xyxy.detach().cpu().numpy()
                    class_ids = result.boxes.cls.detach().cpu().numpy()
                    confidences = result.boxes.conf.detach().cpu().numpy()
                    detector_ids = result.boxes.id.detach().cpu().numpy()

                    for (
                        box,
                        class_id_value,
                        confidence_value,
                        detector_id_value,
                    ) in zip(
                        boxes,
                        class_ids,
                        confidences,
                        detector_ids,
                    ):
                        class_id = int(class_id_value)
                        class_name = (
                            detector.names.get(class_id, str(class_id))
                            if isinstance(detector.names, dict)
                            else detector.names[class_id]
                        )
                        confidence = float(confidence_value)

                        if (
                            str(class_name).lower() != "dog"
                            or confidence < args.conf_threshold
                        ):
                            continue

                        detector_id = int(detector_id_value)
                        detections.append(
                            (
                                np.asarray(box, dtype=np.float32),
                                detector_id,
                            )
                        )
                        confidence_lookup[detector_id] = confidence

                stable_results = stable_tracker.update(
                    detections,
                    source_frame,
                    frame,
                )

                stable_confidence: dict[int, float] = {}
                for detector_id, stable_id in (
                    stable_tracker.yolo_to_stable.items()
                ):
                    if detector_id in confidence_lookup:
                        stable_confidence[int(stable_id)] = (
                            confidence_lookup[detector_id]
                        )

                for box, stable_id in stable_results:
                    x1, y1, x2, y2 = map(float, box)
                    ix1 = int(
                        max(0, min(info["width"], math.floor(x1)))
                    )
                    iy1 = int(
                        max(0, min(info["height"], math.floor(y1)))
                    )
                    ix2 = int(
                        max(0, min(info["width"], math.ceil(x2)))
                    )
                    iy2 = int(
                        max(0, min(info["height"], math.ceil(y2)))
                    )

                    if ix2 <= ix1 or iy2 <= iy1:
                        continue

                    rows.append(
                        {
                            "frame": source_frame,
                            "time_sec": source_frame / source_fps,
                            "track_id": int(stable_id),
                            "x1": ix1,
                            "y1": iy1,
                            "x2": ix2,
                            "y2": iy2,
                            "confidence": stable_confidence.get(
                                int(stable_id), 0.0
                            ),
                            "class_name": "dog",
                        }
                    )

            source_frame += 1

            if source_frame % 500 == 0:
                print(
                    f"Read {source_frame}/{info['frames']} source frames "
                    f"({source_frame / source_fps:.1f}s)"
                )
    finally:
        capture.release()

    tracks = pd.DataFrame(rows, columns=TRACK_COLUMNS)
    temporary_tracks = tracks_path.with_suffix(".csv.tmp")
    tracks.to_csv(temporary_tracks, index=False)
    temporary_tracks.replace(tracks_path)

    elapsed = time.perf_counter() - started
    metadata = {
        "status": "complete",
        "video": file_fingerprint(video),
        "detector_weights": str(args.detector_weights.resolve()),
        "device": args.device,
        "source_fps": source_fps,
        "source_width": int(info["width"]),
        "source_height": int(info["height"]),
        "source_frames_read": int(source_frame),
        "source_duration_seconds": source_frame / source_fps,
        "requested_tracking_fps": float(args.tracking_fps),
        "frame_step": int(frame_step),
        "effective_tracking_fps": float(effective_tracking_fps),
        "sampled_frames": int(sampled_frames),
        "track_rows": int(len(tracks)),
        "unique_track_ids": int(tracks["track_id"].nunique())
        if len(tracks)
        else 0,
        "parameters": {
            "conf_threshold": float(args.conf_threshold),
            "iou_threshold": float(args.iou_threshold),
            "max_age": int(args.max_age),
            "appearance_model": args.appearance_model,
            "appearance_threshold": float(args.appearance_threshold),
            "appearance_ema": float(args.appearance_ema),
            "iou_weight": float(args.iou_weight),
        },
        "elapsed_seconds": float(elapsed),
        "tracks_csv": str(tracks_path),
    }

    temporary_metadata = metadata_path.with_suffix(".json.tmp")
    temporary_metadata.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    temporary_metadata.replace(metadata_path)

    print()
    print("=" * 78)
    print("TRACKING COMPLETE")
    print("=" * 78)
    print("Track rows:", len(tracks))
    print(
        "Unique stable IDs:",
        tracks["track_id"].nunique() if len(tracks) else 0,
    )
    print("Tracks:", tracks_path)
    print("Metadata:", metadata_path)
    print("Elapsed minutes:", round(elapsed / 60.0, 2))


if __name__ == "__main__":
    main()
