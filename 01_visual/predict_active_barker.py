#!/usr/bin/env python3
"""
predict_active_barker.py

Final user-facing active-barker video pipeline.

Examples:
    # Automatically resolve the video and create results/timestamps.csv:
    python predict_active_barker.py \
      --video-id 151 \
      --wav audio.wav \
      --output-dir results

    # Or provide the video while still generating timestamps from video_id:
    python predict_active_barker.py \
      --video-id 151 \
      --video input.mp4 \
      --wav audio.wav \
      --output-dir results

Stages:
1. Resolve video metadata and create timestamps.csv when needed
2. RT-DETR + BoT-SORT + OSNet stable dog tracking
3. Per-bark candidate crop creation
4. ImageBind vision scoring for every candidate track
5. Highest-score selection while preserving every candidate score
6. Annotated output video

Existing tracks.csv is reused only when tracking_metadata.json confirms that
it belongs to the same input video. Use --force-tracking to rerun tracking.
"""

from __future__ import annotations

import os

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_MODEL_DIR = Path(
    os.environ.get(
        "ACTIVE_BARKER_MODEL_DIR",
        "/path/to/models_final",
    )
)

DEFAULT_MODEL = (
    DEFAULT_MODEL_DIR
    / "vision_only_processed_plus_random1000_weight025_final.joblib"
)
DEFAULT_DETECTOR = Path("rtdetr-l.pt")
DEFAULT_RAW_VIDEOS = Path(
    os.environ.get("RAW_VIDEOS_JSON", "/path/to/raw_videos.json")
)
DEFAULT_SENTENCES = Path(
    os.environ.get("SENTENCES_JSON", "/path/to/sentences.json")
)

DEFAULT_VIDEO_ROOT = Path("/path/to/video_root")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete active-barker video pipeline."
    )
    parser.add_argument(
        "--video",
        type=Path,
        help=(
            "Source MP4. Optional with --video-id; when omitted, "
            "the path is resolved from raw_videos.json."
        ),
    )
    parser.add_argument(
        "--video-id",
        help=(
            "Numeric video ID used to look up the video_identifier "
            "and create <output-dir>/timestamps.csv."
        ),
    )
    parser.add_argument(
        "--timestamps",
        type=Path,
        help=(
            "Existing timestamp CSV. When omitted, --video-id is "
            "required and timestamps.csv is generated automatically."
        ),
    )
    parser.add_argument(
        "--wav",
        type=Path,
        help=(
            "Optional full synchronized WAV. When omitted, the pipeline "
            "extracts <output-dir>/full_audio.wav from the resolved video."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--raw-videos-json",
        type=Path,
        default=DEFAULT_RAW_VIDEOS,
    )
    parser.add_argument(
        "--sentences-json",
        type=Path,
        default=DEFAULT_SENTENCES,
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        default=DEFAULT_VIDEO_ROOT,
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
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
    parser.add_argument("--crop-fps", type=float, default=10.0)
    parser.add_argument("--window-stride", type=float, default=1.0)
    parser.add_argument("--padding", type=float, default=0.10)
    parser.add_argument("--output-size", type=int, default=224)
    parser.add_argument("--max-box-gap-sec", type=float, default=1.0)
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--force-tracking", action="store_true")
    parser.add_argument("--skip-annotated-video", action="store_true")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full tracking/scoring logs to the terminal.",
    )
    return parser.parse_args()


def normalize_video_id(value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("video_id is empty")

    try:
        number = float(text)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"video_id must be numeric, received {value!r}"
        ) from error

    if not math.isfinite(number) or not number.is_integer() or number < 0:
        raise ValueError(
            f"video_id must be a non-negative integer, received {value!r}"
        )

    return str(int(number))


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def load_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {path}")

    return [row for row in payload if isinstance(row, dict)]


def find_raw_video_row(
    raw_videos_json: Path,
    video_id: str,
) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []

    for row in load_json_list(raw_videos_json):
        try:
            row_id = normalize_video_id(row.get("video_id"))
        except ValueError:
            continue

        if row_id == video_id:
            matches.append(row)

    if not matches:
        raise KeyError(
            f"video_id {video_id} was not found in {raw_videos_json}"
        )

    if len(matches) > 1:
        print(
            f"Warning: found {len(matches)} raw_videos.json rows for "
            f"video_id {video_id}; using the first."
        )

    return matches[0]


def resolve_video_path(
    raw_row: dict[str, Any],
    video_root: Path,
) -> Path:
    stored_path = str(raw_row.get("video_path", "")).strip()
    if not stored_path:
        raise ValueError("raw_videos.json row has no video_path")

    resolved = (
        video_root / stored_path.lstrip("/")
    ).expanduser().resolve()

    if not resolved.is_file():
        raise FileNotFoundError(
            f"Resolved source video does not exist: {resolved}"
        )

    return resolved


def create_timestamps_csv(
    *,
    video_id: str,
    raw_row: dict[str, Any],
    sentences_json: Path,
    output_path: Path,
) -> tuple[Path, str, int]:
    video_identifier = str(
        raw_row.get("video_identifier", "")
    ).strip()

    if not video_identifier:
        raise ValueError(
            f"raw_videos.json row for video_id {video_id} "
            "has no video_identifier"
        )

    clips: list[tuple[float, float]] = []

    for row in load_json_list(sentences_json):
        identifier = str(
            row.get("video_identifier", "")
        ).strip()

        if identifier != video_identifier:
            continue

        start = finite_float(row.get("start_time"))
        end = finite_float(row.get("end_time"))

        if (
            start is None
            or end is None
            or start < 0
            or end <= start
        ):
            continue

        clips.append((start, end))

    if not clips:
        raise ValueError(
            f"No valid timestamps were found in {sentences_json} "
            f"for video_identifier {video_identifier!r}"
        )

    clips = sorted(set(clips), key=lambda item: (item[0], item[1]))

    rows = [
        {
            "video_id": video_id,
            "video_identifier": video_identifier,
            "start_time": start,
            "end_time": end,
            "audio_file": f"bark_{index:03d}.wav",
            "audio_status": "extract_from_full_wav",
        }
        for index, (start, end) in enumerate(clips)
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(output_path)

    return output_path, video_identifier, len(rows)


def extract_full_wav(
    video: Path,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary = output_path.with_name(
        output_path.stem + ".temporary.wav"
    )
    temporary.unlink(missing_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(video),
        "-vn",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(temporary),
    ]

    result = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            "Could not extract full WAV from the source video:\n"
            + " ".join(command)
            + "\n\n"
            + result.stderr[-4000:]
        )

    if not temporary.is_file() or temporary.stat().st_size == 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"FFmpeg produced no usable WAV from {video}"
        )

    temporary.replace(output_path)
    return output_path


def file_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "absolute_path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def tracking_is_reusable(
    video: Path,
    tracks_path: Path,
    metadata_path: Path,
) -> bool:
    if not tracks_path.is_file() or not metadata_path.is_file():
        return False

    try:
        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )
    except Exception:
        return False

    if metadata.get("status") != "complete":
        return False

    saved = metadata.get("video", {})
    current = file_fingerprint(video)

    return (
        saved.get("absolute_path") == current["absolute_path"]
        and int(saved.get("size_bytes", -1))
        == current["size_bytes"]
        and int(saved.get("mtime_ns", -1))
        == current["mtime_ns"]
    )


def run_logged(
    command: list[str],
    log_path: Path,
    *,
    verbose: bool = False,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None

        for line in process.stdout:
            log.write(line)
            log.flush()

            if verbose:
                print(line, end="")

        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            f"Command failed with exit code {return_code}: "
            + " ".join(command)
            + "\nSee log: "
            + str(log_path)
        )


def write_run_metadata(
    path: Path,
    payload: dict[str, Any],
) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_videos_json = args.raw_videos_json.expanduser().resolve()
    sentences_json = args.sentences_json.expanduser().resolve()
    video_root = args.video_root.expanduser().resolve()

    normalized_video_id: str | None = None
    raw_row: dict[str, Any] | None = None
    generated_timestamps = False
    video_identifier: str | None = None
    timestamp_count: int | None = None

    if args.video_id is not None:
        normalized_video_id = normalize_video_id(args.video_id)
        raw_row = find_raw_video_row(
            raw_videos_json,
            normalized_video_id,
        )

    if args.video is not None:
        video = args.video.expanduser().resolve()
    else:
        if raw_row is None:
            raise ValueError(
                "Provide --video, or provide --video-id so the source "
                "video can be resolved from raw_videos.json."
            )
        video = resolve_video_path(raw_row, video_root)

    if args.timestamps is not None:
        timestamps = args.timestamps.expanduser().resolve()
    else:
        if normalized_video_id is None or raw_row is None:
            raise ValueError(
                "Provide --timestamps, or provide --video-id so "
                "timestamps.csv can be created automatically."
            )

        timestamps = output_dir / "timestamps.csv"
        timestamps, video_identifier, timestamp_count = (
            create_timestamps_csv(
                video_id=normalized_video_id,
                raw_row=raw_row,
                sentences_json=sentences_json,
                output_path=timestamps,
            )
        )
        generated_timestamps = True

        print(
            f"Video {normalized_video_id} ({video_identifier}): "
            f"{timestamp_count} bark event(s)"
        )

    if args.wav is not None:
        wav = args.wav.expanduser().resolve()
        wav_generated = False
    else:
        wav = output_dir / "full_audio.wav"
        print("Preparing full audio...")
        wav = extract_full_wav(video, wav)
        wav_generated = True

    model = args.model.expanduser().resolve()

    # Preserve a bare model name such as "rtdetr-l.pt" so Ultralytics
    # can resolve or download it. Resolve only explicit file paths.
    detector_weights = args.detector_weights.expanduser()
    if detector_weights.parent != Path("."):
        detector_weights = detector_weights.resolve()

    # These inputs must exist locally.
    for path in [video, timestamps, wav, model]:
        if not path.is_file():
            raise FileNotFoundError(path)

    # Explicit detector paths must exist. Bare model names are passed
    # through to Ultralytics for resolution or download.
    if detector_weights.parent != Path(".") and not detector_weights.is_file():
        raise FileNotFoundError(detector_weights)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    script_dir = Path(__file__).resolve().parent
    tracking_script = script_dir / "track_dogs.py"
    scoring_script = script_dir / "score_bark_events.py"

    if not tracking_script.is_file():
        raise FileNotFoundError(tracking_script)
    if not scoring_script.is_file():
        raise FileNotFoundError(scoring_script)

    tracks_path = output_dir / "tracks.csv"
    tracking_metadata = output_dir / "tracking_metadata.json"
    run_metadata_path = output_dir / "run_metadata.json"

    run_started = time.perf_counter()
    stage_timings: dict[str, float] = {}
    tracking_reused = False

    base_metadata: dict[str, Any] = {
        "status": "running",
        "video_id": normalized_video_id,
        "video_identifier": video_identifier,
        "timestamps_generated": generated_timestamps,
        "timestamp_rows": timestamp_count,
        "wav_generated_from_video": wav_generated,
        "inputs": {
            "video": file_fingerprint(video),
            "timestamps": file_fingerprint(timestamps),
            "wav": file_fingerprint(wav),
            "model": file_fingerprint(model),
            "detector_weights": (
                file_fingerprint(detector_weights)
                if detector_weights.is_file()
                else {
                    "model_name": str(detector_weights),
                    "resolved_by_ultralytics": True,
                }
            ),
            "raw_videos_json": (
                file_fingerprint(raw_videos_json)
                if normalized_video_id is not None
                else None
            ),
            "sentences_json": (
                file_fingerprint(sentences_json)
                if generated_timestamps
                else None
            ),
        },
        "output_dir": str(output_dir),
        "tracking_reused": False,
        "stage_timings_seconds": {},
    }
    write_run_metadata(run_metadata_path, base_metadata)

    try:
        should_reuse = (
            not args.force_tracking
            and tracking_is_reusable(
                video,
                tracks_path,
                tracking_metadata,
            )
        )

        if should_reuse:
            tracking_reused = True
            print("[1/3] Tracking: reused")
            stage_timings["tracking"] = 0.0
        else:
            print("[1/3] Tracking: running...")
            tracking_started = time.perf_counter()
            tracking_command = [
                sys.executable,
                str(tracking_script),
                "--video",
                str(video),
                "--output-dir",
                str(output_dir),
                "--detector-weights",
                str(detector_weights),
                "--device",
                args.device,
                "--tracking-fps",
                str(args.tracking_fps),
                "--conf-threshold",
                str(args.conf_threshold),
                "--iou-threshold",
                str(args.iou_threshold),
                "--max-age",
                str(args.max_age),
                "--appearance-model",
                args.appearance_model,
                "--appearance-threshold",
                str(args.appearance_threshold),
                "--appearance-ema",
                str(args.appearance_ema),
                "--iou-weight",
                str(args.iou_weight),
                "--force",
            ]
            run_logged(
                tracking_command,
                logs_dir / "track_dogs.log",
                verbose=args.verbose,
            )
            stage_timings["tracking"] = (
                time.perf_counter() - tracking_started
            )
            print(
                f"[1/3] Tracking: complete "
                f"({stage_timings['tracking'] / 60.0:.2f} min)"
            )

        print("[2/3] Scoring bark events...")
        scoring_started = time.perf_counter()
        scoring_command = [
            sys.executable,
            str(scoring_script),
            "--video",
            str(video),
            "--timestamps",
            str(timestamps),
            "--wav",
            str(wav),
            "--tracks",
            str(tracks_path),
            "--model",
            str(model),
            "--output-dir",
            str(output_dir),
            "--device",
            args.device,
            "--crop-fps",
            str(args.crop_fps),
            "--window-stride",
            str(args.window_stride),
            "--padding",
            str(args.padding),
            "--output-size",
            str(args.output_size),
            "--max-box-gap-sec",
            str(args.max_box_gap_sec),
            "--embedding-batch-size",
            str(args.embedding_batch_size),
        ]
        if args.skip_annotated_video:
            scoring_command.append("--skip-annotated-video")

        run_logged(
            scoring_command,
            logs_dir / "score_bark_events.log",
            verbose=args.verbose,
        )
        stage_timings["scoring"] = (
            time.perf_counter() - scoring_started
        )

        predictions_path = output_dir / "predictions.csv"

        if predictions_path.is_file():
            predictions_df = pd.read_csv(predictions_path)

            for _, row in predictions_df.iterrows():
                event_id = str(row.get("event_id", "event"))
                status = str(row.get("status", "unknown"))

                if status == "predicted":
                    track_id = row.get("predicted_track_id", "")
                    score = float(row.get("top_score", 0.0))

                    print(
                        f"      {event_id}: track {track_id}, "
                        f"score {score:.4f}"
                    )

                elif status == "outside_or_offscreen":
                    print(f"      {event_id}: outside/offscreen")

                else:
                    print(f"      {event_id}: {status}")

        print(
            f"[2/3] Scoring: complete "
            f"({stage_timings['scoring'] / 60.0:.2f} min)"
        )

        if args.skip_annotated_video:
            print("[3/3] Annotated video: skipped")
        else:
            print("[3/3] Annotated video: created")

        total_elapsed = time.perf_counter() - run_started
        final_metadata = {
            **base_metadata,
            "status": "complete",
            "tracking_reused": tracking_reused,
            "stage_timings_seconds": stage_timings,
            "total_elapsed_seconds": float(total_elapsed),
            "outputs": {
                "timestamps_csv": str(timestamps),
                "tracks_csv": str(tracks_path),
                "tracking_metadata": str(tracking_metadata),
                "predictions_csv": str(
                    output_dir / "predictions.csv"
                ),
                "candidate_scores_csv": str(
                    output_dir / "candidate_scores.csv"
                ),
                "annotated_video": (
                    None
                    if args.skip_annotated_video
                    else str(
                        output_dir
                        / "annotated_predictions.mp4"
                    )
                ),
                "crops_root": str(output_dir / "crops"),
                "scoring_metadata": str(
                    output_dir / "scoring_metadata.json"
                ),
                "logs_dir": str(logs_dir),
            },
        }
        write_run_metadata(run_metadata_path, final_metadata)

        print()
        print("=" * 78)
        print("ACTIVE-BARKER PIPELINE COMPLETE")
        print("=" * 78)
        print("Output:", output_dir)
        print("Predictions:", output_dir / "predictions.csv")
        print(
            "Candidate scores:",
            output_dir / "candidate_scores.csv",
        )
        if not args.skip_annotated_video:
            print(
                "Annotated video:",
                output_dir / "annotated_predictions.mp4",
            )
        print("Total minutes:", round(total_elapsed / 60.0, 2))
        print("Detailed logs:", logs_dir)

    except Exception as error:
        total_elapsed = time.perf_counter() - run_started
        failure_metadata = {
            **base_metadata,
            "status": "failed",
            "tracking_reused": tracking_reused,
            "stage_timings_seconds": stage_timings,
            "total_elapsed_seconds": float(total_elapsed),
            "error": f"{type(error).__name__}: {error}",
        }
        write_run_metadata(run_metadata_path, failure_metadata)
        raise


if __name__ == "__main__":
    main()
