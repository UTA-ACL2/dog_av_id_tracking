#!/usr/bin/env python3
"""Run the frozen v2.4.1 all-dog pipeline on all annotated videos.

Default behavior:
- Discover every numeric video directory under inputs/.
- Process all videos with a timestamps.csv and matching source MP4.
- Reuse existing base, profile, and fusion outputs unless --force is used.

The script:
1. Finds each source video and timestamp CSV.
2. Runs predict_multimodal_barker_v2.py only when tracks/predictions are missing.
3. Runs all_dog_profile_manager_v2_4_1.py with the frozen thresholds.
4. Runs fuse_visual_audio_v2.py using the new visual profile mapping.
5. Writes one validation summary CSV.

Existing outputs are reused unless --force is provided.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path('/path/to/outputs')
VIDEO_ROOT = Path('path/to/videos')

DEFAULT_IDS: list[str] = []

DOG2VEC_MODEL = Path(
    "/path/to/checkpoints/dog2vec_130k_9.pt"
)
FAIRSEQ_PATH = Path(
    "/path/to/fairseq"
)
RESNET_CHECKPOINT = Path(
    "/path/to/checkpoints/voxblink_logmel_resnet/best.pt"
)
HYBRID_CHECKPOINT = Path(
    "/path/to/checkpoints/hybrid_fusion_resnet_dog2vec_mfcc/"
    "best_by_eval.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ids",
        nargs="*",
        default=DEFAULT_IDS,
        help=(
            "Optional video IDs. When omitted, discover all numeric IDs "
            "under inputs/ that contain timestamps.csv."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--skip-active-barker",
        action="store_true",
        help=(
            "Do not run predict_multimodal_barker_v2.py. "
            "Require tracks.csv and predictions.csv to already exist."
        ),
    )
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("\nRUNNING:")
    print(" ".join(command))
    subprocess.run(command, check=True)


def find_video(video_id: str) -> Path:
    matches = sorted(VIDEO_ROOT.glob(f"{video_id}_*.mp4"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one video for {video_id}; found {matches}"
        )
    return matches[0]


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def main() -> None:
    args = parse_args()
    python = sys.executable

    summary_rows: list[dict[str, Any]] = []

    if args.ids:
        video_ids = sorted(
            {str(video_id) for video_id in args.ids},
            key=int,
        )
    else:
        video_ids = sorted(
            {
                path.parent.name
                for path in (ROOT / "inputs").glob(
                    "*/timestamps.csv"
                )
                if path.parent.name.isdigit()
            },
            key=int,
        )

    if not video_ids:
        raise RuntimeError("No input videos were discovered.")

    print("Videos to process:", len(video_ids))
    print(video_ids)

    for video_id in video_ids:
        print("\n" + "=" * 78)
        print("PROCESSING", video_id)
        print("=" * 78)

        video = find_video(video_id)
        timestamps = ROOT / "inputs" / video_id / "timestamps.csv"

        if not timestamps.is_file():
            print(f"SKIP {video_id}: missing {timestamps}")
            summary_rows.append(
                {
                    "video_id": video_id,
                    "status": "missing_timestamps",
                }
            )
            continue

        base_output = ROOT / f"v2_results_{video_id}"
        profile_output = ROOT / f"v2_4_1_profiles_{video_id}"
        fused_output = ROOT / f"v2_4_1_fused_{video_id}"

        tracks = base_output / "tracks.csv"
        predictions = base_output / "predictions.csv"
        wav = base_output / "full_audio.wav"

        try:
            if args.force or not (
                tracks.is_file()
                and predictions.is_file()
                and wav.is_file()
            ):
                if args.skip_active_barker:
                    raise FileNotFoundError(
                        "Missing base outputs while --skip-active-barker was set."
                    )

                run(
                    [
                        python,
                        str(ROOT / "predict_multimodal_barker_v2.py"),
                        "--video",
                        str(video),
                        "--timestamps",
                        str(timestamps),
                        "--output-dir",
                        str(base_output),
                        "--dog2vec-model",
                        str(DOG2VEC_MODEL),
                        "--fairseq-path",
                        str(FAIRSEQ_PATH),
                        "--resnet-checkpoint",
                        str(RESNET_CHECKPOINT),
                        "--hybrid-checkpoint",
                        str(HYBRID_CHECKPOINT),
                        "--device",
                        args.device,
                        "--audio-threshold",
                        "0.60",
                        "--visual-similarity-threshold",
                        "0.78",
                        "--save-bark-clips",
                    ]
                )
            else:
                print("Reusing base tracking/scoring outputs:", base_output)

            if args.force and profile_output.exists():
                import shutil
                shutil.rmtree(profile_output)

            if args.force or not (
                profile_output / "track_to_all_dog_profile.csv"
            ).is_file():
                run(
                    [
                        python,
                        str(ROOT / "all_dog_profile_manager_v2_4_1.py"),
                        "--video",
                        str(video),
                        "--tracks",
                        str(tracks),
                        "--output-dir",
                        str(profile_output),
                        "--device",
                        args.device,
                        "--visual-similarity-threshold",
                        "0.65",
                        "--duplicate-iou-threshold",
                        "0.35",
                        "--duplicate-appearance-threshold",
                        "0.75",
                        "--part-duplicate-appearance-threshold",
                        "0.88",
                        "--part-duplicate-min-iou",
                        "0.08",
                        "--part-duplicate-min-intersection-over-smaller",
                        "0.25",
                        "--part-duplicate-max-normalized-center-distance",
                        "0.50",
                        "--intro-outro-window-sec",
                        "15",
                    ]
                )
            else:
                print("Reusing v2.4.1 profile outputs:", profile_output)

            if args.force and fused_output.exists():
                import shutil
                shutil.rmtree(fused_output)
            fused_output.mkdir(parents=True, exist_ok=True)

            if args.force or not (
                fused_output / "bark_to_profile_v2.csv"
            ).is_file():
                run(
                    [
                        python,
                        str(ROOT / "fuse_visual_audio_v2.py"),
                        "--predictions",
                        str(predictions),
                        "--track-profile-map",
                        str(
                            profile_output
                            / "track_to_all_dog_profile.csv"
                        ),
                        "--visual-profiles",
                        str(
                            profile_output
                            / "all_dog_visual_profiles.json"
                        ),
                        "--wav",
                        str(wav),
                        "--output-dir",
                        str(fused_output),
                        "--dog2vec-model",
                        str(DOG2VEC_MODEL),
                        "--fairseq-path",
                        str(FAIRSEQ_PATH),
                        "--resnet-checkpoint",
                        str(RESNET_CHECKPOINT),
                        "--hybrid-checkpoint",
                        str(HYBRID_CHECKPOINT),
                        "--device",
                        args.device,
                        "--audio-threshold",
                        "0.60",
                        "--save-bark-clips",
                    ]
                )
            else:
                print("Reusing fused outputs:", fused_output)

            visual_summary = read_json(
                profile_output / "all_dog_visual_profiles.json"
            )

            mapping = pd.read_csv(
                profile_output / "track_to_all_dog_profile.csv"
            )
            merges = pd.read_csv(
                profile_output / "visual_track_merges.csv"
            )
            quality = pd.read_csv(
                profile_output / "visual_track_quality.csv"
            )

            pipeline_summary_path = (
                fused_output / "pipeline_summary_v2.csv"
            )
            pipeline_summary = (
                pd.read_csv(pipeline_summary_path).iloc[0].to_dict()
                if pipeline_summary_path.is_file()
                else {}
            )

            merge_reason_counts = (
                merges["merge_reason"].value_counts().to_dict()
                if not merges.empty and "merge_reason" in merges.columns
                else {}
            )

            summary_rows.append(
                {
                    "video_id": video_id,
                    "status": "complete",
                    "temporary_tracks": int(
                        visual_summary.get(
                            "temporary_track_ids",
                            quality["track_id"].nunique(),
                        )
                    ),
                    "persistent_visual_profiles": int(
                        mapping["persistent_profile_id"].nunique()
                    ),
                    "track_fragments_reduced": int(
                        visual_summary.get(
                            "track_fragments_merged",
                            quality["track_id"].nunique()
                            - mapping["persistent_profile_id"].nunique(),
                        )
                    ),
                    "part_based_duplicate_merges": int(
                        merge_reason_counts.get(
                            "candidate_part_based_duplicate",
                            0,
                        )
                    ),
                    "fragment_reid_merges": int(
                        merge_reason_counts.get(
                            "candidate_fragment_reid",
                            0,
                        )
                    ),
                    "overlap_duplicate_merges": int(
                        merge_reason_counts.get(
                            "candidate_duplicate_overlap",
                            0,
                        )
                    ),
                    "likely_intro_outro_tracks": int(
                        (
                            quality.get(
                                "track_classification",
                                pd.Series(dtype=str),
                            )
                            == "likely_intro_outro_image"
                        ).sum()
                    ),
                    "uncertain_tracks": int(
                        quality.get(
                            "track_classification",
                            pd.Series(dtype=str),
                        )
                        .isin(
                            [
                                "static_uncertain",
                                "likely_intro_outro_image",
                            ]
                        )
                        .sum()
                    ),
                    "total_persistent_dogs": pipeline_summary.get(
                        "total_persistent_dogs"
                    ),
                    "barking_dogs": pipeline_summary.get(
                        "barking_dogs"
                    ),
                    "silent_dogs": pipeline_summary.get("silent_dogs"),
                    "bark_events": pipeline_summary.get("bark_events"),
                    "assigned_bark_events": pipeline_summary.get(
                        "assigned_bark_events"
                    ),
                    "audio_visual_conflicts": pipeline_summary.get(
                        "audio_visual_conflicts"
                    ),
                    "base_output": str(base_output),
                    "profile_output": str(profile_output),
                    "fused_output": str(fused_output),
                }
            )

        except Exception as error:
            print(f"FAILED {video_id}: {error}")
            summary_rows.append(
                {
                    "video_id": video_id,
                    "status": "failed",
                    "error": str(error),
                }
            )

    summary = pd.DataFrame(summary_rows)
    summary_path = ROOT / "v2_4_1_all_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 78)
    print("ALL SUMMARY")
    print("=" * 78)

    display_columns = [
        "video_id",
        "status",
        "temporary_tracks",
        "persistent_visual_profiles",
        "track_fragments_reduced",
        "part_based_duplicate_merges",
        "fragment_reid_merges",
        "likely_intro_outro_tracks",
        "total_persistent_dogs",
        "barking_dogs",
        "silent_dogs",
        "audio_visual_conflicts",
    ]
    available = [
        column for column in display_columns
        if column in summary.columns
    ]
    print(summary[available].to_string(index=False))

    complete = summary[summary["status"] == "complete"].copy()
    if not complete.empty:
        numeric_columns = [
            "temporary_tracks",
            "persistent_visual_profiles",
            "track_fragments_reduced",
            "part_based_duplicate_merges",
            "fragment_reid_merges",
            "overlap_duplicate_merges",
            "likely_intro_outro_tracks",
            "uncertain_tracks",
            "total_persistent_dogs",
            "barking_dogs",
            "silent_dogs",
            "bark_events",
            "assigned_bark_events",
            "audio_visual_conflicts",
        ]

        totals = {
            column: pd.to_numeric(
                complete[column],
                errors="coerce",
            ).sum()
            for column in numeric_columns
            if column in complete.columns
        }

        aggregate = pd.DataFrame(
            [
                {
                    "videos_complete": len(complete),
                    "videos_failed": int(
                        (summary["status"] == "failed").sum()
                    ),
                    "videos_missing_timestamps": int(
                        (
                            summary["status"]
                            == "missing_timestamps"
                        ).sum()
                    ),
                    **totals,
                    "mean_temporary_tracks_per_video": (
                        pd.to_numeric(
                            complete["temporary_tracks"],
                            errors="coerce",
                        ).mean()
                        if "temporary_tracks" in complete
                        else float("nan")
                    ),
                    "mean_persistent_profiles_per_video": (
                        pd.to_numeric(
                            complete["persistent_visual_profiles"],
                            errors="coerce",
                        ).mean()
                        if "persistent_visual_profiles" in complete
                        else float("nan")
                    ),
                    "mean_profile_reduction_per_video": (
                        pd.to_numeric(
                            complete["track_fragments_reduced"],
                            errors="coerce",
                        ).mean()
                        if "track_fragments_reduced" in complete
                        else float("nan")
                    ),
                    "overall_profile_reduction_fraction": (
                        totals.get("track_fragments_reduced", 0)
                        / totals.get("temporary_tracks", 1)
                        if totals.get("temporary_tracks", 0)
                        else float("nan")
                    ),
                    "bark_assignment_rate": (
                        totals.get("assigned_bark_events", 0)
                        / totals.get("bark_events", 1)
                        if totals.get("bark_events", 0)
                        else float("nan")
                    ),
                    "audio_visual_conflict_rate": (
                        totals.get("audio_visual_conflicts", 0)
                        / totals.get("assigned_bark_events", 1)
                        if totals.get("assigned_bark_events", 0)
                        else float("nan")
                    ),
                }
            ]
        )

        aggregate_path = (
            ROOT / "v2_4_1_all_aggregate.csv"
        )
        aggregate.to_csv(aggregate_path, index=False)

        print("\nAggregate results:")
        print(aggregate.to_string(index=False))
        print("Saved:", aggregate_path)

    print("\nSaved:", summary_path)


if __name__ == "__main__":
    main()
