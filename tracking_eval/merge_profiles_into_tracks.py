from pathlib import Path
import pandas as pd


RAW_DIR = Path("raw-outputs")
MERGED_DIR = Path("merged-outputs")


def merge_video(video_id):

    tracks_file = (
        RAW_DIR /
        video_id /
        "tracks.csv"
    )

    mapping_file = (
        MERGED_DIR /
        video_id /
        "track_to_all_dog_profile.csv"
    )


    if not tracks_file.exists():
        print(
            f"{video_id}: missing tracks"
        )
        return


    if not mapping_file.exists():
        print(
            f"{video_id}: missing mapping"
        )
        return


    tracks = pd.read_csv(
        tracks_file
    )

    mapping = pd.read_csv(
        mapping_file
    )


    merged = tracks.merge(
        mapping[
            [
                "track_id",
                "persistent_profile_id"
            ]
        ],
        on="track_id",
        how="left"
    )


    # sanity check
    missing = merged.persistent_profile_id.isna().sum()

    if missing:
        print(
            f"{video_id}: {missing} detections "
            "have no profile"
        )


    output = (
        MERGED_DIR /
        video_id /
        "tracks_with_profiles.csv"
    )


    merged.to_csv(
        output,
        index=False
    )


    print(
        f"Saved {output}"
    )



def main():

    videos = sorted(
        [
            x.name
            for x in MERGED_DIR.iterdir()
            if x.is_dir()
        ]
    )


    for video_id in videos:
        merge_video(video_id)



if __name__ == "__main__":
    main()