from pathlib import Path
import argparse
import subprocess
import sys


DATASET_DIR = Path("dataset")
RAW_OUTPUT_DIR = Path("raw-outputs")
MERGED_OUTPUT_DIR = Path("merged-outputs")

PROFILE_SCRIPT = "all_dog_profile_manager.py"


def run_profile_manager(video, tracks_csv, output_dir, device):

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    done_file = (
        output_dir /
        "track_to_all_dog_profile.csv"
    )

    if done_file.exists():
        print(
            f"Skipping {video.stem} (already merged)"
        )
        return


    cmd = [
        sys.executable,
        PROFILE_SCRIPT,

        "--video",
        str(video),

        "--tracks",
        str(tracks_csv),

        "--output-dir",
        str(output_dir),

        "--device",
        device,
    ]


    print("\n" + "="*80)
    print("Running:")
    print(" ".join(cmd))
    print("="*80)


    try:
        subprocess.run(
            cmd,
            check=True
        )

        print(
            f"Finished {video.stem}"
        )

    except subprocess.CalledProcessError:
        print(
            f"FAILED {video.stem}"
        )



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Passed through to all_dog_profile_manager.py, e.g. cuda:0 or cpu",
    )
    args = parser.parse_args()

    videos = sorted(
        DATASET_DIR.glob("*.mp4"),
        key=lambda x: int(x.stem)
        if x.stem.isdigit()
        else x.stem
    )


    for video in videos:

        video_id = video.stem


        tracks_csv = (
            RAW_OUTPUT_DIR /
            video_id /
            "tracks.csv"
        )


        if not tracks_csv.exists():

            print(
                f"Skipping {video_id}: "
                "missing tracks.csv"
            )

            continue


        output_dir = (
            MERGED_OUTPUT_DIR /
            video_id
        )


        run_profile_manager(
            video,
            tracks_csv,
            output_dir,
            args.device
        )


if __name__ == "__main__":
    main()