#!/usr/bin/env python3

import argparse
import subprocess
import sys
import time
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent


def build_stages(device: str):
    return [
        (
            "1. Convert CVAT XML -> MOT ground truth",
            [sys.executable, "cvat_to_trackeval.py"]
        ),

        (
            "2. Run full-frame detection + tracking",
            [sys.executable, "run_all_tracking.py", "--device", device]
        ),

        (
            "3. Build persistent dog profiles",
            [sys.executable, "predictions_to_mot.py", "--device", device]
        ),

        (
            "4. Merge profiles back into tracks",
            [sys.executable, "merge_profiles_into_tracks.py"]
        ),

        (
            "5. Convert tracks -> TrackEval MOT format",
            [sys.executable, "tracks_to_trackeval.py"]
        ),

        (
            "6. Run baseline trackers",
            [
                sys.executable,
                "run_alt_trackers.py",
                "--videos-dir", "dataset",
                "--root", "MOT_eval",
                "--device", device,
            ]
        ),

        (
            "7. Run TrackEval metrics",
            [
                sys.executable,
                "run_mot_eval.py",
                "--root", "MOT_eval",
                "--only-sampled-frames",
                "--out", "MOT_eval/eval_results.csv"
            ]
        ),
    ]


# ============================================================
# RUNNER
# ============================================================

def run_stage(name, command):
    print("\n" + "=" * 90)
    print(name)
    print("=" * 90)

    start = time.time()

    try:
        subprocess.run(
            command,
            cwd=ROOT,
            check=True
        )

    except subprocess.CalledProcessError as e:
        print("\n❌ FAILED:")
        print(" ".join(command))
        print(f"Exit code: {e.returncode}")

        sys.exit(e.returncode)

    elapsed = time.time() - start

    print("\n✅ Finished")
    print(f"Time: {elapsed/60:.2f} min")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Forwarded to every stage that runs the detector/ReID model, "
        "e.g. cuda:0 (default) or cpu",
    )
    args = parser.parse_args()

    print(f"""
============================================================

 DOG TRACKING EVALUATION PIPELINE
 device: {args.device}

 1. CVAT -> MOT GT
 2. Detection + Tracking
 3. Dog Profile Generation
 4. Profile Merge
 5. TrackEval Conversion
 6. Baseline Trackers
 7. Evaluation

============================================================
""")

    for name, command in build_stages(args.device):
        run_stage(name, command)

    print("\n\n🎉 PIPELINE COMPLETE")
    print("Results:")
    print(ROOT / "MOT_eval/eval_results.csv")


if __name__ == "__main__":
    main()