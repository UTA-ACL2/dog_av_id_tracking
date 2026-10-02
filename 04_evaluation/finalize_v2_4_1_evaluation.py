#!/usr/bin/env python3
"""Finalize the v2.4.1 evaluation from existing pipeline outputs.

This script does not rerun detection, tracking, profiling, or fusion. It reads
the completed result folders and creates a clean final evaluation package.

Expected inputs per video:
- v2_results_<ID>/tracks.csv
- v2_4_1_profiles_<ID>/track_to_all_dog_profile.csv
- v2_4_1_profiles_<ID>/visual_track_merges.csv
- v2_4_1_profiles_<ID>/visual_track_quality.csv
- v2_4_1_fused_<ID>/pipeline_summary_v2.csv
- v2_4_1_fused_<ID>/bark_to_profile_v2.csv

Videos with empty timestamp files are treated as visual-only, not failed.

Outputs:
- final_v2_4_1_evaluation/per_video_results.csv
- final_v2_4_1_evaluation/aggregate_results.csv
- final_v2_4_1_evaluation/merge_breakdown.csv
- final_v2_4_1_evaluation/audio_visual_conflicts.csv
- final_v2_4_1_evaluation/paper_ready_results.txt
- final_v2_4_1_evaluation/final_results.xlsx
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path('/path/to/outputs')
INPUT_ROOT = ROOT / "inputs"
OUTPUT_ROOT = ROOT / "final_v2_4_1_evaluation"


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def safe_read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def discover_video_ids() -> list[str]:
    ids = {
        path.parent.name
        for path in INPUT_ROOT.glob("*/timestamps.csv")
        if path.parent.name.isdigit()
    }
    return sorted(ids, key=int)


def timestamp_event_count(path: Path) -> int:
    df = safe_read_csv(path)
    return len(df)


def count_conflicts(assignments: pd.DataFrame) -> int:
    if assignments.empty:
        return 0

    if "audio_visual_agreement" in assignments.columns:
        values = assignments["audio_visual_agreement"]
        normalized = values.astype(str).str.lower()
        return int(normalized.isin(["false", "0"]).sum())

    if "v2_decision" in assignments.columns:
        return int(
            (
                assignments["v2_decision"].astype(str)
                == "audio_visual_identity_conflict"
            ).sum()
        )

    return 0


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    video_ids = discover_video_ids()
    rows: list[dict[str, Any]] = []
    conflict_frames: list[pd.DataFrame] = []

    for video_id in video_ids:
        timestamp_path = (
            INPUT_ROOT / video_id / "timestamps.csv"
        )
        annotated_bark_events = timestamp_event_count(timestamp_path)

        base_dir = ROOT / f"v2_results_{video_id}"
        profile_dir = ROOT / f"v2_4_1_profiles_{video_id}"
        fused_dir = ROOT / f"v2_4_1_fused_{video_id}"

        tracks = safe_read_csv(base_dir / "tracks.csv")
        mapping = safe_read_csv(
            profile_dir / "track_to_all_dog_profile.csv"
        )
        merges = safe_read_csv(
            profile_dir / "visual_track_merges.csv"
        )
        quality = safe_read_csv(
            profile_dir / "visual_track_quality.csv"
        )
        assignments = safe_read_csv(
            fused_dir / "bark_to_profile_v2.csv"
        )
        fused_summary = safe_read_csv(
            fused_dir / "pipeline_summary_v2.csv"
        )
        visual_json = safe_read_json(
            profile_dir / "all_dog_visual_profiles.json"
        )

        if tracks.empty or mapping.empty:
            rows.append(
                {
                    "video_id": video_id,
                    "evaluation_status": "missing_visual_outputs",
                    "annotated_bark_events": annotated_bark_events,
                }
            )
            continue

        temporary_tracks = int(tracks["track_id"].nunique())
        persistent_profiles = int(
            mapping["persistent_profile_id"].nunique()
        )
        profile_reduction = temporary_tracks - persistent_profiles

        merge_counts = (
            merges["merge_reason"].value_counts().to_dict()
            if not merges.empty and "merge_reason" in merges.columns
            else {}
        )

        likely_intro_outro = 0
        uncertain_tracks = 0
        if not quality.empty and "track_classification" in quality.columns:
            classifications = quality["track_classification"].astype(str)
            likely_intro_outro = int(
                (
                    classifications
                    == "likely_intro_outro_image"
                ).sum()
            )
            uncertain_tracks = int(
                classifications.isin(
                    [
                        "static_uncertain",
                        "likely_intro_outro_image",
                    ]
                ).sum()
            )

        if annotated_bark_events == 0:
            evaluation_status = "visual_complete_no_bark_events"
        elif fused_summary.empty:
            evaluation_status = "visual_complete_missing_fusion"
        else:
            evaluation_status = "complete_multimodal"

        fused = (
            fused_summary.iloc[0].to_dict()
            if not fused_summary.empty
            else {}
        )

        assigned_events = int(
            fused.get(
                "assigned_bark_events",
                len(assignments),
            )
            or 0
        )
        bark_events = int(
            fused.get(
                "bark_events",
                annotated_bark_events,
            )
            or 0
        )
        conflicts = int(
            fused.get(
                "audio_visual_conflicts",
                count_conflicts(assignments),
            )
            or 0
        )

        if not assignments.empty:
            conflict_mask = pd.Series(False, index=assignments.index)

            if "audio_visual_agreement" in assignments.columns:
                normalized = (
                    assignments["audio_visual_agreement"]
                    .astype(str)
                    .str.lower()
                )
                conflict_mask |= normalized.isin(["false", "0"])

            if "v2_decision" in assignments.columns:
                conflict_mask |= (
                    assignments["v2_decision"].astype(str)
                    == "audio_visual_identity_conflict"
                )

            conflicts_df = assignments[conflict_mask].copy()
            if not conflicts_df.empty:
                conflicts_df = conflicts_df.copy()
                conflicts_df["video_id"] = video_id

                ordered_columns = [
                    "video_id",
                    *[
                        column
                        for column in conflicts_df.columns
                        if column != "video_id"
                    ],
                ]
                conflicts_df = conflicts_df[ordered_columns]
                conflict_frames.append(conflicts_df)

        rows.append(
            {
                "video_id": video_id,
                "evaluation_status": evaluation_status,
                "annotated_bark_events": annotated_bark_events,
                "temporary_tracks": temporary_tracks,
                "persistent_visual_profiles": persistent_profiles,
                "track_fragments_reduced": profile_reduction,
                "profile_reduction_fraction": (
                    profile_reduction / temporary_tracks
                    if temporary_tracks > 0
                    else np.nan
                ),
                "part_based_duplicate_merges": int(
                    merge_counts.get(
                        "candidate_part_based_duplicate",
                        0,
                    )
                ),
                "fragment_reid_merges": int(
                    merge_counts.get(
                        "candidate_fragment_reid",
                        0,
                    )
                ),
                "overlap_duplicate_merges": int(
                    merge_counts.get(
                        "candidate_duplicate_overlap",
                        0,
                    )
                ),
                "likely_intro_outro_tracks": likely_intro_outro,
                "uncertain_tracks": uncertain_tracks,
                "total_persistent_dogs": fused.get(
                    "total_persistent_dogs",
                    persistent_profiles,
                ),
                "barking_dogs": fused.get("barking_dogs", np.nan),
                "silent_dogs": fused.get("silent_dogs", np.nan),
                "bark_events": bark_events,
                "assigned_bark_events": assigned_events,
                "audio_visual_conflicts": conflicts,
                "bark_assignment_rate": (
                    assigned_events / bark_events
                    if bark_events > 0
                    else np.nan
                ),
                "audio_visual_conflict_rate": (
                    conflicts / assigned_events
                    if assigned_events > 0
                    else np.nan
                ),
                "visual_profile_version": visual_json.get(
                    "version",
                    "v2.4.1",
                ),
            }
        )

    per_video = pd.DataFrame(rows).sort_values(
        "video_id",
        key=lambda series: series.astype(int),
    )

    visual_complete = per_video[
        per_video["evaluation_status"].isin(
            [
                "complete_multimodal",
                "visual_complete_no_bark_events",
            ]
        )
    ].copy()

    multimodal_complete = per_video[
        per_video["evaluation_status"] == "complete_multimodal"
    ].copy()

    temporary_total = int(
        pd.to_numeric(
            visual_complete["temporary_tracks"],
            errors="coerce",
        ).sum()
    )
    persistent_total = int(
        pd.to_numeric(
            visual_complete["persistent_visual_profiles"],
            errors="coerce",
        ).sum()
    )
    reduction_total = temporary_total - persistent_total

    bark_total = int(
        pd.to_numeric(
            multimodal_complete["bark_events"],
            errors="coerce",
        ).sum()
    )
    assigned_total = int(
        pd.to_numeric(
            multimodal_complete["assigned_bark_events"],
            errors="coerce",
        ).sum()
    )
    conflict_total = int(
        pd.to_numeric(
            multimodal_complete["audio_visual_conflicts"],
            errors="coerce",
        ).sum()
    )

    aggregate = pd.DataFrame(
        [
            {
                "videos_discovered": len(per_video),
                "videos_visual_complete": len(visual_complete),
                "videos_multimodal_complete": len(multimodal_complete),
                "videos_visual_only_no_barks": int(
                    (
                        per_video["evaluation_status"]
                        == "visual_complete_no_bark_events"
                    ).sum()
                ),
                "videos_missing_visual_outputs": int(
                    (
                        per_video["evaluation_status"]
                        == "missing_visual_outputs"
                    ).sum()
                ),
                "videos_missing_fusion": int(
                    (
                        per_video["evaluation_status"]
                        == "visual_complete_missing_fusion"
                    ).sum()
                ),
                "temporary_tracks": temporary_total,
                "persistent_visual_profiles": persistent_total,
                "track_fragments_reduced": reduction_total,
                "overall_profile_reduction_fraction": (
                    reduction_total / temporary_total
                    if temporary_total > 0
                    else np.nan
                ),
                "mean_temporary_tracks_per_video": (
                    temporary_total / len(visual_complete)
                    if len(visual_complete) > 0
                    else np.nan
                ),
                "mean_persistent_profiles_per_video": (
                    persistent_total / len(visual_complete)
                    if len(visual_complete) > 0
                    else np.nan
                ),
                "mean_profile_reduction_per_video": (
                    reduction_total / len(visual_complete)
                    if len(visual_complete) > 0
                    else np.nan
                ),
                "part_based_duplicate_merges": int(
                    pd.to_numeric(
                        visual_complete[
                            "part_based_duplicate_merges"
                        ],
                        errors="coerce",
                    ).sum()
                ),
                "fragment_reid_merges": int(
                    pd.to_numeric(
                        visual_complete["fragment_reid_merges"],
                        errors="coerce",
                    ).sum()
                ),
                "overlap_duplicate_merges": int(
                    pd.to_numeric(
                        visual_complete[
                            "overlap_duplicate_merges"
                        ],
                        errors="coerce",
                    ).sum()
                ),
                "likely_intro_outro_tracks": int(
                    pd.to_numeric(
                        visual_complete[
                            "likely_intro_outro_tracks"
                        ],
                        errors="coerce",
                    ).sum()
                ),
                "uncertain_tracks": int(
                    pd.to_numeric(
                        visual_complete["uncertain_tracks"],
                        errors="coerce",
                    ).sum()
                ),
                "bark_events": bark_total,
                "assigned_bark_events": assigned_total,
                "bark_assignment_rate": (
                    assigned_total / bark_total
                    if bark_total > 0
                    else np.nan
                ),
                "audio_visual_conflicts": conflict_total,
                "audio_visual_conflict_rate": (
                    conflict_total / assigned_total
                    if assigned_total > 0
                    else np.nan
                ),
            }
        ]
    )

    merge_breakdown = pd.DataFrame(
        [
            {
                "merge_type": "fragment_reidentification",
                "count": int(
                    aggregate.iloc[0]["fragment_reid_merges"]
                ),
            },
            {
                "merge_type": "part_based_duplicate",
                "count": int(
                    aggregate.iloc[0][
                        "part_based_duplicate_merges"
                    ]
                ),
            },
            {
                "merge_type": "overlap_duplicate",
                "count": int(
                    aggregate.iloc[0][
                        "overlap_duplicate_merges"
                    ]
                ),
            },
        ]
    )
    merge_total = int(merge_breakdown["count"].sum())
    merge_breakdown["fraction_of_accepted_merges"] = (
        merge_breakdown["count"] / merge_total
        if merge_total > 0
        else np.nan
    )

    conflicts_all = (
        pd.concat(conflict_frames, ignore_index=True)
        if conflict_frames
        else pd.DataFrame()
    )

    per_video_path = OUTPUT_ROOT / "per_video_results.csv"
    aggregate_path = OUTPUT_ROOT / "aggregate_results.csv"
    merge_path = OUTPUT_ROOT / "merge_breakdown.csv"
    conflicts_path = OUTPUT_ROOT / "audio_visual_conflicts.csv"
    workbook_path = OUTPUT_ROOT / "final_results.xlsx"
    text_path = OUTPUT_ROOT / "paper_ready_results.txt"

    per_video.to_csv(per_video_path, index=False)
    aggregate.to_csv(aggregate_path, index=False)
    merge_breakdown.to_csv(merge_path, index=False)
    conflicts_all.to_csv(conflicts_path, index=False)

    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        per_video.to_excel(
            writer,
            sheet_name="Per Video",
            index=False,
        )
        aggregate.to_excel(
            writer,
            sheet_name="Aggregate",
            index=False,
        )
        merge_breakdown.to_excel(
            writer,
            sheet_name="Merge Breakdown",
            index=False,
        )
        conflicts_all.to_excel(
            writer,
            sheet_name="AV Conflicts",
            index=False,
        )

    result = aggregate.iloc[0]

    paper_text = f"""FINAL V2.4.1 EVALUATION

Dataset coverage
- Videos processed visually: {int(result['videos_visual_complete'])}
- Videos evaluated multimodally: {int(result['videos_multimodal_complete'])}
- Visual-only videos with no annotated bark events: {int(result['videos_visual_only_no_barks'])}

Visual identity consolidation
- Temporary tracker IDs: {int(result['temporary_tracks'])}
- Persistent visual profiles: {int(result['persistent_visual_profiles'])}
- Track fragments reduced: {int(result['track_fragments_reduced'])}
- Overall profile reduction: {100 * result['overall_profile_reduction_fraction']:.2f}%
- Mean temporary tracks per video: {result['mean_temporary_tracks_per_video']:.2f}
- Mean persistent profiles per video: {result['mean_persistent_profiles_per_video']:.2f}
- Mean profile reduction per video: {result['mean_profile_reduction_per_video']:.2f}

Accepted merge categories
- Fragment re-identification merges: {int(result['fragment_reid_merges'])}
- Part-based duplicate merges: {int(result['part_based_duplicate_merges'])}
- Overlap duplicate merges: {int(result['overlap_duplicate_merges'])}
- Likely intro/outro tracks: {int(result['likely_intro_outro_tracks'])}
- Uncertain visual tracks: {int(result['uncertain_tracks'])}

Multimodal bark assignment
- Bark events: {int(result['bark_events'])}
- Assigned bark events: {int(result['assigned_bark_events'])}
- Bark assignment rate: {100 * result['bark_assignment_rate']:.2f}%
- Audio-visual conflicts: {int(result['audio_visual_conflicts'])}
- Audio-visual conflict rate: {100 * result['audio_visual_conflict_rate']:.2f}%

Paper-ready summary
Across {int(result['videos_visual_complete'])} annotated videos, the visual
identity manager consolidated {int(result['temporary_tracks'])} temporary
tracker IDs into {int(result['persistent_visual_profiles'])} persistent visual
profiles, corresponding to a {100 * result['overall_profile_reduction_fraction']:.1f}%
reduction in tracker fragmentation. Fragment re-identification accounted for
{int(result['fragment_reid_merges'])} accepted merges, while part-based and
overlapping duplicate handling contributed
{int(result['part_based_duplicate_merges'])} and
{int(result['overlap_duplicate_merges'])} merges, respectively. Multimodal bark
assignment was evaluated on {int(result['videos_multimodal_complete'])} videos
containing annotated bark events. The pipeline assigned
{int(result['assigned_bark_events'])} of {int(result['bark_events'])} bark events
to persistent profiles, achieving a
{100 * result['bark_assignment_rate']:.1f}% assignment rate. Audio and visual
identity evidence disagreed for {int(result['audio_visual_conflicts'])} assigned
events ({100 * result['audio_visual_conflict_rate']:.1f}%).

Interpretation note
Profile reduction measures consolidation of tracker fragments and duplicate
tracks. It should not be described as physical-identity accuracy because the
dataset does not provide complete ground-truth physical dog identities for every
video. Likewise, bark assignment rate measures coverage, not correctness of the
assigned physical identity.
"""
    text_path.write_text(paper_text, encoding="utf-8")

    print("\nFINAL V2.4.1 EVALUATION")
    print("=" * 72)
    print(aggregate.to_string(index=False))
    print("\nMerge breakdown:")
    print(merge_breakdown.to_string(index=False))

    print("\nSaved:")
    print(per_video_path)
    print(aggregate_path)
    print(merge_path)
    print(conflicts_path)
    print(workbook_path)
    print(text_path)


if __name__ == "__main__":
    main()
