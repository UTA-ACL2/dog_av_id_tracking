#!/usr/bin/env python3
"""Attach bark events to pre-existing all-dog visual profiles.

Visible bark events are attached to the persistent profile associated with the
predicted visual track. Audio updates that profile and provides conflict
signals. Off-screen events use audio-only matching and may create an
``audio_only_dog_N`` profile when no existing barking profile matches.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from audio_fingerprint import AudioFingerprintEncoder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--track-profile-map", type=Path, required=True)
    p.add_argument("--visual-profiles", type=Path, required=True)
    p.add_argument("--wav", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dog2vec-model", type=Path, required=True)
    p.add_argument("--fairseq-path", type=Path, required=True)
    p.add_argument("--resnet-checkpoint", type=Path, required=True)
    p.add_argument("--hybrid-checkpoint", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--audio-threshold", type=float, default=0.60)
    p.add_argument("--min-duration", type=float, default=0.25)
    p.add_argument("--max-duration", type=float, default=5.0)
    p.add_argument("--save-bark-clips", action="store_true")
    return p.parse_args()


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))


def update_centroid(old: np.ndarray, new: np.ndarray, count: int) -> np.ndarray:
    centroid = (old * count + new) / (count + 1)
    return centroid / (np.linalg.norm(centroid) + 1e-12)


def main() -> None:
    args = parse_args()
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(args.predictions.expanduser().resolve()).sort_values(
        ["start_time_sec", "end_time_sec", "event_id"]
    ).reset_index(drop=True)
    mapping = pd.read_csv(args.track_profile_map.expanduser().resolve())
    track_to_profile = {
        int(row.track_id): str(row.persistent_profile_id)
        for row in mapping.itertuples(index=False)
    }
    profile_data = json.loads(args.visual_profiles.expanduser().resolve().read_text())
    profiles: dict[str, dict[str, Any]] = profile_data["profiles"]

    full_audio, sr = librosa.load(args.wav.expanduser().resolve(), sr=16000, mono=True)
    encoder = AudioFingerprintEncoder(
        dog2vec_model=args.dog2vec_model,
        fairseq_path=args.fairseq_path,
        resnet_checkpoint=args.resnet_checkpoint,
        hybrid_checkpoint=args.hybrid_checkpoint,
        device=args.device,
        sample_rate=sr,
    )

    audio_centroids: dict[str, np.ndarray] = {}
    audio_counts: dict[str, int] = {}
    next_audio_only = 1
    bark_dir = out / "v2_bark_clips"
    if args.save_bark_clips:
        bark_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for index, row in predictions.iterrows():
        base = row.to_dict()
        event_id = str(row["event_id"])
        start, end = float(row["start_time_sec"]), float(row["end_time_sec"])
        duration = end - start
        track_value = finite_float(row.get("predicted_track_id"))
        track_id = int(track_value) if track_value is not None else None
        visual_profile = track_to_profile.get(track_id) if track_id is not None else None

        base.update({
            "all_dog_profile_id": visual_profile or "",
            "assignment_source": "",
            "audio_best_profile_id": "",
            "audio_similarity": "",
            "audio_visual_agreement": "",
            "v2_decision": "",
            "audio_clip_status": "",
            "audio_clip_path": "",
        })

        if duration < args.min_duration or duration > args.max_duration:
            if visual_profile:
                assigned = visual_profile
                decision = "visible_profile_audio_not_scored"
                source = "visual"
            else:
                assigned = ""
                decision = "unassigned_audio_not_scored"
                source = "none"
            base.update({"all_dog_profile_id": assigned, "assignment_source": source,
                         "v2_decision": decision,
                         "audio_clip_status": f"invalid_duration_{duration:.3f}"})
            if assigned:
                profiles[assigned]["bark_events"].append(event_id)
                profiles[assigned]["bark_count"] += 1
                profiles[assigned]["status"] = "barking"
            rows.append(base)
            continue

        ss = max(0, int(round(start * sr)))
        es = min(len(full_audio), int(round(end * sr)))
        clip = full_audio[ss:es]
        if len(clip) == 0:
            base.update({"assignment_source": "visual" if visual_profile else "none",
                         "v2_decision": "empty_audio_clip",
                         "audio_clip_status": "empty_clip"})
            rows.append(base)
            continue

        embedding = encoder.encode(clip)
        scores = {pid: cosine(embedding, centroid) for pid, centroid in audio_centroids.items()}
        best_audio_profile = max(scores, key=scores.get) if scores else None
        best_audio_score = float(scores[best_audio_profile]) if best_audio_profile else None

        if visual_profile:
            assigned = visual_profile
            source = "visual_plus_audio"
            if visual_profile in audio_centroids:
                own_score = cosine(embedding, audio_centroids[visual_profile])
            else:
                own_score = None
            if best_audio_profile and best_audio_profile != visual_profile and best_audio_score is not None and best_audio_score >= args.audio_threshold:
                decision = "audio_visual_identity_conflict"
                agreement = False
            else:
                decision = "attached_to_visible_all_dog_profile"
                agreement = True if best_audio_profile in {None, visual_profile} else False
            count = audio_counts.get(assigned, 0)
            if count == 0:
                audio_centroids[assigned] = embedding
                audio_counts[assigned] = 1
            else:
                audio_centroids[assigned] = update_centroid(audio_centroids[assigned], embedding, count)
                audio_counts[assigned] = count + 1
            reported_similarity = own_score if own_score is not None else best_audio_score
        else:
            if best_audio_profile is not None and best_audio_score is not None and best_audio_score >= args.audio_threshold:
                assigned = best_audio_profile
                decision = "offscreen_matched_existing_profile_by_audio"
                source = "audio"
                agreement = True
                count = audio_counts[assigned]
                audio_centroids[assigned] = update_centroid(audio_centroids[assigned], embedding, count)
                audio_counts[assigned] = count + 1
            else:
                assigned = f"audio_only_dog_{next_audio_only}"
                next_audio_only += 1
                profiles[assigned] = {
                    "track_ids": [], "first_seen_time": start, "last_seen_time": end,
                    "track_fragments": 0, "visual_embedding_available": False,
                    "bark_events": [], "bark_count": 0, "status": "barking_offscreen_only",
                }
                audio_centroids[assigned] = embedding
                audio_counts[assigned] = 1
                decision = "created_offscreen_audio_only_profile"
                source = "audio"
                agreement = ""
            reported_similarity = best_audio_score

        profiles[assigned]["bark_events"].append(event_id)
        profiles[assigned]["bark_count"] += 1
        if profiles[assigned]["track_ids"]:
            profiles[assigned]["status"] = "barking"

        clip_path = ""
        if args.save_bark_clips:
            destination = bark_dir / f"{index:04d}_{event_id}_{start:.3f}_{end:.3f}.wav"
            sf.write(destination, clip, sr)
            clip_path = str(destination)

        base.update({
            "all_dog_profile_id": assigned,
            "assignment_source": source,
            "audio_best_profile_id": best_audio_profile or "",
            "audio_similarity": reported_similarity if reported_similarity is not None else "",
            "audio_visual_agreement": agreement,
            "v2_decision": decision,
            "audio_clip_status": "ok",
            "audio_clip_path": clip_path,
        })
        rows.append(base)

    output = pd.DataFrame(rows)
    output.to_csv(out / "bark_to_profile_v2.csv", index=False)

    for pid, profile in profiles.items():
        profile["audio_count"] = audio_counts.get(pid, 0)
    final = {
        "version": "v2",
        "audio_threshold": args.audio_threshold,
        "profiles": profiles,
        "summary": {
            "total_persistent_dogs": len(profiles),
            "barking_dogs": sum(int(p["bark_count"] > 0) for p in profiles.values()),
            "silent_dogs": sum(int(p["bark_count"] == 0) for p in profiles.values()),
            "visual_profiles": sum(int(bool(p["track_ids"])) for p in profiles.values()),
            "offscreen_audio_only_profiles": sum(int(not p["track_ids"]) for p in profiles.values()),
            "bark_events": len(output),
            "assigned_bark_events": int(output["all_dog_profile_id"].fillna("").astype(str).ne("").sum()),
            "audio_visual_conflicts": int(output["v2_decision"].eq("audio_visual_identity_conflict").sum()),
        },
    }
    (out / "all_dog_profiles_v2.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    pd.DataFrame([final["summary"]]).to_csv(out / "pipeline_summary_v2.csv", index=False)

    print("Bark assignments:", out / "bark_to_profile_v2.csv")
    print("Profiles:", out / "all_dog_profiles_v2.json")
    print(pd.DataFrame([final["summary"]]).to_string(index=False))


if __name__ == "__main__":
    main()
