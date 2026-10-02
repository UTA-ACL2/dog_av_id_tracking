from pathlib import Path
import pandas as pd


RAW_DIR = Path("raw-outputs")
MERGED_DIR = Path("merged-outputs")

OUT_DIR = Path("MOT_eval/trackers")


def convert_csv(
    csv_path,
    output_path,
    id_column
):

    df = pd.read_csv(csv_path)

    required = [
        "frame",
        id_column,
        "x1",
        "y1",
        "x2",
        "y2",
        "confidence"
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        print(
            f"Skipping {csv_path}: missing {missing}"
        )
        return


    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    with open(output_path, "w") as f:

        for _, row in df.iterrows():

            frame = int(row["frame"])

            obj_id = row[id_column]

            # TrackEval expects:
            # x,y,width,height

            x = float(row["x1"])
            y = float(row["y1"])

            w = float(row["x2"] - row["x1"])
            h = float(row["y2"] - row["y1"])

            conf = float(row["confidence"])


            f.write(
                f"{frame},{obj_id},"
                f"{x},{y},{w},{h},"
                f"{conf},-1,-1\n"
            )


def process_video(video_id):

    print(f"\nProcessing {video_id}")

    # -----------------------
    # raw tracker
    # -----------------------

    raw_csv = (
        RAW_DIR /
        video_id /
        "tracks.csv"
    )


    if raw_csv.exists():

        out = (
            OUT_DIR /
            "raw_tracker" /
            "data" /
            f"{video_id}.txt"
        )

        convert_csv(
            raw_csv,
            out,
            "track_id"
        )

        print(
            f"raw -> {out}"
        )


    else:
        print(
            f"missing {raw_csv}"
        )


    # -----------------------
    # profile tracker
    # -----------------------

    profile_csv = (
        MERGED_DIR /
        video_id /
        "tracks_with_profiles.csv"
    )


    if profile_csv.exists():

        out = (
            OUT_DIR /
            "profile_tracker" /
            "data" /
            f"{video_id}.txt"
        )


        convert_csv(
            profile_csv,
            out,
            "persistent_profile_id"
        )


        print(
            f"profile -> {out}"
        )

    else:
        print(
            f"missing {profile_csv}"
        )


def main():

    videos = sorted(
        [
            x.name
            for x in RAW_DIR.iterdir()
            if x.is_dir()
        ]
    )


    for video_id in videos:
        process_video(video_id)



if __name__ == "__main__":
    main()