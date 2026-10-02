from pathlib import Path
import argparse
import subprocess
import sys


DATASET_DIR = Path("dataset")
OUTPUT_DIR = Path("raw-outputs")

TRACK_SCRIPT = "track_dogs.py"


def run_tracker(video_path: Path, output_dir: Path, device: str):

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    tracks_file = output_dir / "tracks.csv"

    # resume support
    if tracks_file.exists():
        print(
            f"Skipping {video_path.name} (already done)"
        )
        return


    cmd = [
        sys.executable,
        TRACK_SCRIPT,

        "--video",
        str(video_path),

        "--output-dir",
        str(output_dir),

        "--device",
        device,
    ]


    print("\n" + "=" * 80)
    print("Running:")
    print(" ".join(cmd))
    print("=" * 80)


    try:
        subprocess.run(
            cmd,
            check=True
        )

        print(
            f"Finished {video_path.name}"
        )

    except subprocess.CalledProcessError as e:

        print(
            f"FAILED {video_path.name}"
        )
        print(e)



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Passed through to track_dogs.py, e.g. cuda:0 or cpu",
    )
    args = parser.parse_args()

    videos = sorted(
        DATASET_DIR.glob("*.mp4")
    )


    if not videos:
        raise RuntimeError(
            f"No mp4 files found in {DATASET_DIR}"
        )


    print(
        f"Found {len(videos)} videos"
    )


    for video in videos:

        video_id = video.stem

        output_dir = (
            OUTPUT_DIR / video_id
        )

        run_tracker(
            video,
            output_dir,
            args.device
        )


if __name__ == "__main__":
    main()