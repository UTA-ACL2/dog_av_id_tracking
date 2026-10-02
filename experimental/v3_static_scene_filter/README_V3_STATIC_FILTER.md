# V3 Static Intro/Outro Dog-Graphic Filter

## Purpose

This experimental preprocessing stage removes dog detections that come from static intro cards, outro screens, thumbnails, collages, or channel graphics before persistent visual profiles are created.

It was added after manual review showed that several one-dog videos were overcounted because the outro contained multiple still images of dogs and each still image received its own temporary track.

## Current Status

The frozen paper baseline remains:

```text
all_dog_profile_manager_v2_4_1.py
fuse_visual_audio_v2.py
finalize_v2_4_1_evaluation.py
```

The v3 filter is experimental and should be evaluated separately.

## Main Script

```text
filter_static_dog_graphics_v3.py
```

## Processing Order

```text
Original tracks.csv
    ↓
V3 static-scene filter
    ↓
tracks_live_scene_v3.csv
    ↓
all_dog_profile_manager_v2_4_1.py
    ↓
persistent live-dog profiles
```

## Default Parameters

```text
boundary window:                         20 seconds
analysis step:                           0.5 seconds
static motion threshold:                 3.0
minimum static interval:                 3.0 seconds
minimum simultaneous dog tracks:         2
track static fraction threshold:         0.80
maximum scene motion for removal:        4.0
scene-cut difference threshold:          18.0
```

## Run on Video 2135

```bash
cd /path/to/multimodal_pipeline
conda activate dog2vec_gpu

rm -rf /path/to/multimodal_pipeline/outputs/v3_static_filter_VIDEO_ID

python filter_static_dog_graphics_v3.py   --video /path/to/videos/VIDEO_ID.mp4   --tracks /path/to/outputs/v2_results_VIDEO_ID/tracks.csv   --output-dir /path/to/outputs/v3_static_filter_VIDEO_ID   --boundary-window-sec 20   --analysis-step-sec 0.5   --static-motion-threshold 3.0   --min-static-duration-sec 3.0   --min-simultaneous-dog-tracks 2   --track-static-fraction-threshold 0.80
```

## Inspect Decisions

```bash
column -s, -t < /path/to/outputs/v3_static_filter_VIDEO_ID/static_scene_intervals.csv
```

```bash
column -s, -t < /path/to/outputs/v3_static_filter_VIDEO_ID/static_scene_track_decisions.csv
```

```bash
cat /path/to/outputs/v3_static_filter_VIDEO_ID/static_scene_summary.json
```

## Run Persistent Profiling on Filtered Tracks

```bash
rm -rf /path/to/outputs/v3_static_filter_VIDEO_ID/

python all_dog_profile_manager_v2_4_1.py   --video /path/to/videos/VIDEO_ID.mp4   --tracks /path/to/outputs/v3_static_filter_VIDEO_ID/tracks_live_scene_v3.csv   --output-dir /path/to/outputs/v3_profiles_VIDEO_ID   --device cuda:0   --visual-similarity-threshold 0.65   --duplicate-iou-threshold 0.35   --duplicate-appearance-threshold 0.75   --part-duplicate-appearance-threshold 0.88   --part-duplicate-min-iou 0.08   --part-duplicate-min-intersection-over-smaller 0.25   --part-duplicate-max-normalized-center-distance 0.50   --intro-outro-window-sec 15
```

## Main Outputs

```text
tracks_live_scene_v3.csv
boundary_scene_samples.csv
static_scene_intervals.csv
static_scene_track_decisions.csv
static_scene_summary.json
review_frames/
```

## Expected Result

For videos such as 723:

```text
static dog-image tracks removed
real dog tracks retained
persistent dog count reduced
```

This filter does not solve stuffed animals, identity switches, one box covering multiple dogs, or real-dog fragments that fail to merge.
