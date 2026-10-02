# Batch CVAT Evaluation

This batch evaluator runs the spatial active-barker evaluator across all videos
that have top-level CVAT annotations and completed pipeline outputs.

## Valid Metrics

- Active-barker localization accuracy
- Mean spatial IoU
- Per-video bark event results
- Optional manual dog-count accuracy

It intentionally does not aggregate CVAT track IDs as persistent dog identities,
because the CVAT tracks may correspond to bark events rather than unique dogs.

## Run

```bash
cd ~/multimodal_pipeline/FINAL_MODEL_RELEASE
conda activate dog2vec_gpu

python 04_evaluation/batch_evaluate_cvat_localization.py   --annotations-root /path/to/annotations   --videos-root /path/to/videos   --pipeline-output-root /path/to/outputs/cvat_evaluation_runs   --output-dir /path/to/outputs/cvat_batch_results   --evaluator 04_evaluation/evaluate_cvat_barker_identity.py   --spatial-iou-threshold 0.30   --force
```

## Optional Manual Counts

Create `manual_dog_counts.csv`:

```csv
video_id,total_dogs,barking_dogs,silent_dogs
80461,3,1,2
```

Then add:

```bash
--manual-counts 04_evaluation/manual_dog_counts.csv
```

## Required Existing Outputs

For each video ID:

```text
<pipeline-output-root>/<video_id>_base/predictions.csv
<pipeline-output-root>/<video_id>_base/tracks.csv
<pipeline-output-root>/<video_id>_profiles/track_to_all_dog_profile.csv
<pipeline-output-root>/<video_id>_fusion/pipeline_summary_v2.csv
```

## Outputs

```text
per_video_cvat_results.csv
aggregate_cvat_results.csv
batch_failures.csv
batch_summary.txt
<video_id>/
    bark_event_localization.csv
    aggregate_metrics.csv
    ...
```
