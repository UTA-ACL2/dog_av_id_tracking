#!/usr/bin/env python3
"""Run the v2 all-dog pipeline while preserving the original pipeline."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--timestamps", type=Path, required=True)
    p.add_argument("--wav", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dog2vec-model", type=Path, required=True)
    p.add_argument("--fairseq-path", type=Path, required=True)
    p.add_argument("--resnet-checkpoint", type=Path, required=True)
    p.add_argument("--hybrid-checkpoint", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--audio-threshold", type=float, default=0.60)
    p.add_argument("--visual-similarity-threshold", type=float, default=0.78)
    p.add_argument("--force", action="store_true")
    p.add_argument("--save-bark-clips", action="store_true")
    p.add_argument("--skip-annotated-video", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def run(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    a = parse_args()
    root = Path(__file__).resolve().parent
    out = a.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    visual = [sys.executable, str(root / "predict_active_barker.py"),
              "--video", str(a.video), "--timestamps", str(a.timestamps),
              "--output-dir", str(out), "--device", a.device]
    if a.wav:
        visual += ["--wav", str(a.wav)]
    if a.force:
        visual.append("--force-tracking")
    if a.skip_annotated_video:
        visual.append("--skip-annotated-video")
    if a.verbose:
        visual.append("--verbose")
    run(visual)

    all_dog = [sys.executable, str(root / "all_dog_profile_manager_v2.py"),
               "--video", str(a.video), "--tracks", str(out / "tracks.csv"),
               "--output-dir", str(out), "--device", a.device,
               "--visual-similarity-threshold", str(a.visual_similarity_threshold)]
    if a.force:
        all_dog.append("--force")
    run(all_dog)

    wav = a.wav.expanduser().resolve() if a.wav else out / "full_audio.wav"
    fusion = [sys.executable, str(root / "fuse_visual_audio_v2.py"),
              "--predictions", str(out / "predictions.csv"),
              "--track-profile-map", str(out / "track_to_all_dog_profile.csv"),
              "--visual-profiles", str(out / "all_dog_visual_profiles.json"),
              "--wav", str(wav), "--output-dir", str(out),
              "--dog2vec-model", str(a.dog2vec_model),
              "--fairseq-path", str(a.fairseq_path),
              "--resnet-checkpoint", str(a.resnet_checkpoint),
              "--hybrid-checkpoint", str(a.hybrid_checkpoint),
              "--device", a.device, "--audio-threshold", str(a.audio_threshold)]
    if a.save_bark_clips:
        fusion.append("--save-bark-clips")
    run(fusion)


if __name__ == "__main__":
    main()
