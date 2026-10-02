# CVAT Active-Barker Evaluation

This evaluator compares the pipeline against full-video CVAT XML annotations.

## Required Inputs

- Original video
- Top-level `annotations.xml` for the full video
- `predictions.csv` from the active-barker stage
- `tracks.csv`
- `track_to_all_dog_profile.csv`

## Example: Video 80461

```bash
cd ~/multimodal_pipeline/FINAL_MODEL_RELEASE
conda activate dog2vec_gpu

python 04_evaluation/evaluate_cvat_barker_identity.py \
  --video /path/to/videos/80461_*.mp4 \
  --annotations-xml /path/to/annotations/80461/annotations.xml \
  --predictions /path/to/80461_base/predictions.csv \
  --tracks /path/to/80461_base/tracks.csv \
  --track-profile-map /path/to/80461_profiles/track_to_all_dog_profile.csv \
  --output-dir /path/to/80461_cvat_evaluation \
  --spatial-iou-threshold 0.30
```

Shell wildcards should be resolved before passing the video path. A safer approach is:

```bash
VIDEO=$(find /path/to/videos \
  -maxdepth 1 -name "80461_*.mp4" | head -n 1)
```

Then use `--video "$VIDEO"`.

## Primary Metric

`active_barker_localization_accuracy`

An event is correct when the predicted temporary-track box overlaps the manually annotated barking-dog box with spatial IoU at or above the threshold.

## Identity Metrics

The script uses Hungarian alignment between persistent profile IDs and CVAT track IDs.

These metrics are valid only when CVAT track IDs are persistent identities. If the annotation process created a new track for every bark event, use only the spatial localization metrics.

## Outputs

- `bark_event_localization.csv`
- `profile_identity_mapping.csv`
- `profile_purity.csv`
- `gt_track_fragmentation.csv`
- `cvat_track_summary.csv`
- `aggregate_metrics.csv`
- `evaluation_summary.txt`
