#!/usr/bin/env python3
"""
smoke_test.py

End-to-end sanity check for the dog-tracking pipeline, run in an isolated
temp directory so it never touches your real dataset/annotations/outputs.

Always runs (no GPU / torch / ultralytics / torchreid required):
    - cvat_to_trackeval.py       (CVAT XML -> MOT gt)
    - merge_profiles_into_tracks.py
    - tracks_to_trackeval.py
    - run_mot_eval.py            (real TrackEval HOTA/MOTA/IDF1 computation)
using small synthetic fixture files, to prove the "glue" logic (parsing,
merging, format conversion, metric computation) is wired together correctly.

Additionally runs, ONLY if torch + ultralytics + torchreid are importable
(pip install -r requirements.txt):
    - run_all_tracking.py --device cpu   (real RT-DETR detection + tracking
      on a tiny synthetic video)
    - predictions_to_mot.py --device cpu (real OSNet-based profile building)
    - run_alt_trackers.py --device cpu   (SORT/ByteTrack/BoT-SORT baselines)
using a generated 2-second synthetic video. This exercises the real model
code paths on CPU. It will very likely find zero dogs (the synthetic video
has no real dog in it) - that's expected and fine. The point is to prove
nothing crashes, imports resolve, and files land where later stages expect.

Usage:
    python smoke_test.py               # fixture-only glue test
    python smoke_test.py --full        # also attempt the model stages (slow, CPU)
    python smoke_test.py --full --keep # keep the temp dir for inspection
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent

REQUIRED_SCRIPTS = [
    "cvat_to_trackeval.py",
    "merge_profiles_into_tracks.py",
    "tracks_to_trackeval.py",
    "run_mot_eval.py",
    "run_all_tracking.py",
    "predictions_to_mot.py",
    "run_alt_trackers.py",
    "track_dogs.py",
    "all_dog_profile_manager.py",
]


def check(cond, msg):
    print(("[PASS] " if cond else "[FAIL] ") + msg)
    if not cond:
        raise SystemExit(1)


def run(cmd, cwd):
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def write_fixtures(work: Path):
    (work / "annotations").mkdir(parents=True, exist_ok=True)
    (work / "dataset").mkdir(parents=True, exist_ok=True)
    (work / "raw-outputs" / "1").mkdir(parents=True, exist_ok=True)
    (work / "merged-outputs" / "1").mkdir(parents=True, exist_ok=True)

    (work / "annotations" / "1.xml").write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <track id="0" label="dog">
    <box frame="0" xtl="10" ytl="10" xbr="60" ybr="60" outside="0"/>
    <box frame="1" xtl="12" ytl="10" xbr="62" ybr="60" outside="0"/>
    <box frame="2" xtl="14" ytl="10" xbr="64" ybr="60" outside="0"/>
    <box frame="3" xtl="16" ytl="10" xbr="66" ybr="60" outside="0"/>
    <box frame="4" xtl="18" ytl="10" xbr="68" ybr="60" outside="0"/>
  </track>
</annotations>
"""
    )

    (work / "raw-outputs" / "1" / "tracks.csv").write_text(
        "frame,track_id,x1,y1,x2,y2,confidence\n"
        "1,7,11,10,61,60,0.91\n"
        "2,7,13,10,63,60,0.92\n"
        "3,7,15,10,65,60,0.90\n"
        "4,7,17,10,67,60,0.88\n"
        "5,7,19,10,69,60,0.89\n"
    )

    (work / "merged-outputs" / "1" / "track_to_all_dog_profile.csv").write_text(
        "track_id,persistent_profile_id\n7,dog_A\n"
    )


def glue_test(work: Path):
    print("\n" + "=" * 70)
    print("GLUE TEST (no GPU / torch required)")
    print("=" * 70)

    write_fixtures(work)

    run([sys.executable, "cvat_to_trackeval.py"], cwd=work)
    check((work / "MOT_eval" / "gt" / "1.txt").exists(), "gt/1.txt created")

    run([sys.executable, "merge_profiles_into_tracks.py"], cwd=work)
    check(
        (work / "merged-outputs" / "1" / "tracks_with_profiles.csv").exists(),
        "tracks_with_profiles.csv created",
    )

    run([sys.executable, "tracks_to_trackeval.py"], cwd=work)
    check(
        (work / "MOT_eval" / "trackers" / "raw_tracker" / "data" / "1.txt").exists(),
        "raw_tracker/data/1.txt created",
    )
    check(
        (work / "MOT_eval" / "trackers" / "profile_tracker" / "data" / "1.txt").exists(),
        "profile_tracker/data/1.txt created",
    )

    run(
        [
            sys.executable,
            "run_mot_eval.py",
            "--root",
            "MOT_eval",
            "--only-sampled-frames",
            "--out",
            "MOT_eval/eval_results.csv",
        ],
        cwd=work,
    )
    check((work / "MOT_eval" / "eval_results.csv").exists(), "eval_results.csv written")

    print("\nGlue test passed: CVAT parsing, profile merge, MOT conversion, "
          "and TrackEval scoring all ran successfully.")


def make_synthetic_video(path: Path, n_frames=48, size=(320, 240), fps=24):
    import cv2
    import numpy as np

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    for i in range(n_frames):
        frame = np.full((size[1], size[0], 3), 40, dtype=np.uint8)
        x = 10 + i * 3
        cv2.rectangle(frame, (x, 80), (x + 60, 140), (0, 200, 0), -1)
        writer.write(frame)
    writer.release()


def model_stage_deps_available():
    try:
        import torch  # noqa: F401
        import torchreid  # noqa: F401
        import ultralytics  # noqa: F401
        return True
    except ImportError as e:
        print(f"\n[SKIP] model-stage test: {e}")
        print("Install requirements.txt to enable --full model-stage testing.")
        return False


def model_stage_test(work: Path):
    print("\n" + "=" * 70)
    print("MODEL-STAGE TEST (CPU, requires torch/ultralytics/torchreid)")
    print("=" * 70)

    weights = ROOT / "rtdetr-l.pt"
    if not weights.exists():
        print(f"[SKIP] {weights} not found next to this script.")
        return

    (work / "dataset").mkdir(parents=True, exist_ok=True)
    make_synthetic_video(work / "dataset" / "1.mp4")

    run(
        [sys.executable, "run_all_tracking.py", "--device", "cpu"],
        cwd=work,
    )
    check((work / "raw-outputs" / "1" / "tracks.csv").exists(), "real tracks.csv produced")

    run(
        [sys.executable, "predictions_to_mot.py", "--device", "cpu"],
        cwd=work,
    )

    run(
        [
            sys.executable,
            "run_alt_trackers.py",
            "--videos-dir",
            "dataset",
            "--root",
            "MOT_eval",
            "--device",
            "cpu",
            "--methods",
            "sort",
        ],
        cwd=work,
    )

    print("\nModel-stage test passed: detector + tracker + baseline code ran "
          "on CPU without errors (0 dog detections is expected - the "
          "synthetic video has no real dog in it).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Also run the GPU/model stages on CPU")
    parser.add_argument("--keep", action="store_true", help="Keep the temp working directory")
    args = parser.parse_args()

    missing = [s for s in REQUIRED_SCRIPTS if not (ROOT / s).exists()]
    check(not missing, f"all pipeline scripts present next to smoke_test.py (missing: {missing})")

    work_dir = Path(tempfile.mkdtemp(prefix="dogtrack_smoke_"))
    print(f"Working directory: {work_dir}")

    # copy every pipeline script into the sandboxed working dir so relative
    # imports (e.g. run_alt_trackers.py importing track_dogs.py) resolve
    for f in ROOT.glob("*.py"):
        shutil.copy(f, work_dir / f.name)

    try:
        glue_test(work_dir)

        if args.full:
            if model_stage_deps_available():
                model_stage_test(work_dir)
        else:
            print(
                "\n(Skipping model stages - pass --full to also exercise "
                "run_all_tracking.py / predictions_to_mot.py / "
                "run_alt_trackers.py on CPU with a synthetic video.)"
            )

        print("\nALL SMOKE TESTS PASSED")
    finally:
        if args.keep:
            print(f"\nKept working directory for inspection: {work_dir}")
        else:
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
