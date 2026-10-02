#!/usr/bin/env python3
"""One-command wrapper: run visual active-barker pipeline, then audio fusion."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the combined multimodal pipeline.")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--video-id")
    parser.add_argument("--timestamps", type=Path)
    parser.add_argument("--wav", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dog2vec-model", type=Path, required=True)
    parser.add_argument("--fairseq-path", type=Path, required=True)
    parser.add_argument("--resnet-checkpoint", type=Path, required=True)
    parser.add_argument("--hybrid-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--similarity-threshold", type=float, default=0.60)
    parser.add_argument("--force-tracking", action="store_true")
    parser.add_argument("--skip-annotated-video", action="store_true")
    parser.add_argument("--save-bark-clips", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    visual = [
        sys.executable,
        str(script_dir / "predict_active_barker.py"),
        "--output-dir",
        str(output_dir),
        "--device",
        args.device,
    ]
    if args.video is not None:
        visual += ["--video", str(args.video)]
    if args.video_id is not None:
        visual += ["--video-id", str(args.video_id)]
    if args.timestamps is not None:
        visual += ["--timestamps", str(args.timestamps)]
    if args.wav is not None:
        visual += ["--wav", str(args.wav)]
    if args.force_tracking:
        visual.append("--force-tracking")
    if args.skip_annotated_video:
        visual.append("--skip-annotated-video")
    if args.verbose:
        visual.append("--verbose")
    run(visual)

    resolved_wav = args.wav.expanduser().resolve() if args.wav else output_dir / "full_audio.wav"
    fusion = [
        sys.executable,
        str(script_dir / "fuse_visual_audio.py"),
        "--predictions",
        str(output_dir / "predictions.csv"),
        "--wav",
        str(resolved_wav),
        "--output-dir",
        str(output_dir),
        "--dog2vec-model",
        str(args.dog2vec_model),
        "--fairseq-path",
        str(args.fairseq_path),
        "--resnet-checkpoint",
        str(args.resnet_checkpoint),
        "--hybrid-checkpoint",
        str(args.hybrid_checkpoint),
        "--device",
        args.device,
        "--similarity-threshold",
        str(args.similarity_threshold),
    ]
    if args.save_bark_clips:
        fusion.append("--save-bark-clips")
    run(fusion)


if __name__ == "__main__":
    main()
