#!/usr/bin/env python3

from pathlib import Path
from collections import defaultdict
import argparse
import pandas as pd


def analyze_gt(gt_dir):

    gt_dir = Path(gt_dir)

    all_frames = set()
    total_boxes = 0
    all_tracks = set()

    per_video = []

    for gt_file in sorted(gt_dir.glob("*.txt")):

        video_id = gt_file.stem

        rows = []

        with open(gt_file, "r") as f:
            for line in f:
                if not line.strip():
                    continue

                parts = line.strip().split(",")

                if len(parts) < 6:
                    continue

                frame = int(parts[0])
                track_id = int(parts[1])

                rows.append((frame, track_id))

        if not rows:
            continue

        df = pd.DataFrame(
            rows,
            columns=["frame", "track_id"]
        )

        frames = df["frame"].nunique()
        boxes = len(df)
        tracks = df["track_id"].nunique()

        all_frames.update(df["frame"].unique())
        all_tracks.update(
            (video_id, t)
            for t in df["track_id"].unique()
        )

        track_lengths = (
            df.groupby("track_id")
            .size()
        )

        per_video.append({
            "video": video_id,
            "frames": frames,
            "boxes": boxes,
            "tracks": tracks,
            "avg_track_length": track_lengths.mean(),
            "max_track_length": track_lengths.max(),
        })


    result = pd.DataFrame(per_video)


    print("\n==============================")
    print("GROUND TRUTH SUMMARY")
    print("==============================")

    print(f"Videos annotated: {len(result)}")
    print(f"Total unique frames: {len(all_frames)}")
    print(f"Total boxes: {sum(result.boxes)}")
    print(f"Total individual tracks: {len(all_tracks)}")

    print("\n------------------------------")
    print("Per-video statistics")
    print("------------------------------")

    print(result.to_string(index=False))


    print("\n------------------------------")
    print("Averages")
    print("------------------------------")

    print(
        f"Frames/video: {result.frames.mean():.2f}"
    )

    print(
        f"Boxes/video: {result.boxes.mean():.2f}"
    )

    print(
        f"Tracks/video: {result.tracks.mean():.2f}"
    )

    print(
        f"Boxes/frame: {(result.boxes.sum()/result.frames.sum()):.2f}"
    )


    result.to_csv(
        "gt_statistics.csv",
        index=False
    )

    print("\nSaved:")
    print("gt_statistics.csv")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gt",
        default="MOT_eval/gt",
        help="Path to MOT ground truth folder"
    )

    args = parser.parse_args()

    analyze_gt(args.gt)