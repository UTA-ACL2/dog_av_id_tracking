# Multimodal Dog Fingerprinting Pipeline

This repository accompanies the paper "Audio-Visual Dog Identification and Tracking in Real-World Videos".

The pipeline combines visual tracking and canine vocal fingerprinting to identify individual dogs in unconstrained videos, maintain persistent identities across the video, and associate barking events with the correct dog.

Unlike traditional tracking pipelines, this system is designed specifically for unconstrained YouTube videos where dogs may become occluded, leave and re-enter the camera, multiple dogs may bark simultaneously, and videos often contain intro/outro graphics that resemble real dogs.

---

# Pipeline Overview

The complete pipeline is shown below.

```
Input Video
      │
      ▼
RT-DETR Dog Detection
      │
      ▼
Temporary Multi-Object Tracking
      │
      ▼
Static Intro/Outro Scene Filtering (v3)
      │
      ▼
Track Quality Analysis
      │
      ▼
Appearance Embedding Extraction (OSNet)
      │
      ▼
Duplicate & Fragment Merging
      │
      ▼
Persistent Visual Dog Profiles
      │
      ▼
VideoMAE Active Bark Prediction
      │
      ▼
Audio Fingerprinting
      │
      ▼
Visual-Audio Fusion
      │
      ▼
Final Persistent Dog Profiles
```

The final output consists of persistent dog identities throughout the video together with barking events assigned to each dog.

---

# Directory Structure

```
multimodal_pipeline/

predict_multimodal_barker_v2.py

filter_static_dog_graphics_v3.py

all_dog_profile_manager_v2_4_1.py

fuse_visual_audio_v2.py

run_v3_static_all_annotations.py

split_identity_switches_v2_5.py

README_FINAL_MULTIMODAL_PIPELINE.md
```

---

# Pipeline Components

## 1. Dog Detection

Model

```
RT-DETR
```

Purpose

Detect every visible dog in every frame.

Output

```
tracks.csv
```

This stage generates temporary tracker IDs that are later refined into persistent identities.

---

## 2. Temporary Tracking

Purpose

Maintain a temporary identity while the dog remains visible.

Outputs

```
tracks.csv

tracking_metadata.json
```

Limitations

- identity switches
- tracker fragmentation
- merged dogs during occlusion

These are handled later by the profile manager.

---

## 3. Static Scene Filtering (v3)

Script

```
filter_static_dog_graphics_v3.py
```

Purpose

Many YouTube videos contain

- intro cards
- end screens
- slideshow graphics
- static dog pictures

These images frequently generate false tracker IDs that are interpreted as additional dogs.

The static-scene filter detects these regions before profile construction.

Detection criteria include

- very low frame-to-frame motion
- long stationary duration
- boundary scene (intro/outro)
- multiple simultaneous stationary dog tracks

Outputs

```
tracks_live_scene_v3.csv

static_scene_intervals.csv

static_scene_track_decisions.csv

static_scene_summary.json
```

Tracks identified as static graphics are removed before profile generation.

---

## 4. Track Quality Analysis

Purpose

Compute quality statistics for every temporary track.

Metrics

- duration
- observation density
- internal motion
- total movement
- box size
- sparse detection
- static detection

Outputs

```
visual_track_quality.csv
```

Tracks are classified as

```
active_real_candidate

sparse_candidate

likely_intro_outro_image
```

---

## 5. Appearance Embedding Extraction

Model

```
OSNet-x0.25
```

Purpose

Extract appearance descriptors for every track.

For each track we compute

- appearance embedding
- running average embedding

These embeddings are later used for

- fragment merging
- duplicate removal
- visual profile construction

---

## 6. Duplicate & Fragment Merging

Script

```
all_dog_profile_manager_v2_4_1.py
```

Purpose

Convert temporary tracker IDs into persistent dog identities.

Three merge strategies are used.

### Fragment Re-identification

Reconnect tracks after

- leaving camera
- re-entering
- temporary occlusion

Uses

- appearance similarity
- temporal gap

---

### Part-Based Duplicate Merge

Detect cases where two trackers follow different parts of the same dog.

Uses

- appearance similarity
- IoU
- intersection-over-smaller
- normalized center distance

---

### Overlap Duplicate Merge

Detect overlapping duplicate trackers.

Uses

- overlap statistics
- appearance similarity

Outputs

```
track_to_all_dog_profile.csv

visual_track_merges.csv

visual_track_pair_diagnostics.csv

all_dog_visual_profiles.json
```

---

## 7. Active Bark Prediction

Model

```
VideoMAE
```

Purpose

Determine which visible dog is actively barking during each bark event.

Output

```
predictions.csv
```

---

## 8. Audio Fingerprinting

Audio features

- MFCC
- Dog2Vec
- Log-Mel Spectrogram

The three feature representations are fused into a single 256-dimensional embedding using the hybrid fingerprint network.

Each bark clip is embedded independently.

Outputs

```
audio fingerprint embeddings
```

---

## 9. Visual-Audio Fusion

Script

```
fuse_visual_audio_v2.py
```

Purpose

Combine

- visual profiles
- bark timing
- audio fingerprints

The system creates

- barking dog profiles
- silent dog profiles
- off-screen audio-only profiles (if necessary)

Outputs

```
pipeline_summary_v2.csv

bark_to_profile_v2.csv

all_dog_profiles_v2.json
```

---

# Final Outputs

For every video the pipeline generates

```
Persistent visual profiles

Barking dog profiles

Silent dog profiles

Assigned bark events

Audio-visual conflict report

Visual profile mapping
```

---

# Running the Pipeline

Activate environment

```bash
conda activate dog2vec_gpu
```

Move into project

```bash
cd ~/multimodal_pipeline
```

---

## Step 1

Generate temporary tracks

```
predict_multimodal_barker_v2.py
```

---

## Step 2

Remove intro/outro graphics

```
filter_static_dog_graphics_v3.py
```

---

## Step 3

Generate persistent visual profiles

```
all_dog_profile_manager_v2_4_1.py
```

---

## Step 4

Run multimodal fusion

```
fuse_visual_audio_v2.py
```

---

## Step 5

Evaluate all videos

```
run_v3_static_all_annotations.py
```

Outputs

```
v3_static_all_annotations_summary.csv

v3_static_all_annotations_aggregate.csv

v2_4_1_vs_v3_static_comparison.csv

v2_4_1_vs_v3_static_aggregate.csv
```

---

# Final Evaluation

Videos processed

```
23
```

Videos with bark events

```
22
```

Temporary tracker IDs

```
148
```

Static graphic tracks removed

```
3
```

Filtered temporary tracks

```
145
```

Persistent visual profiles

```
97
```

Track reduction

```
51
```

Overall profile reduction

```
34.46%
```

Fragment ReID merges

```
38
```

Part-based duplicate merges

```
8
```

Overlap duplicate merges

```
2
```

Bark events

```
109
```

Assigned bark events

```
108
```

Bark assignment rate

```
99.08%
```

Audio-visual conflicts

```
14
```

Conflict rate

```
12.96%
```

---

# Remaining Limitations

The remaining failures are primarily caused by limitations of the underlying tracker rather than the multimodal profile manager.

Common failure cases include

- severe occlusion
- identity switches
- multiple dogs merged into one detection
- fragmented tracks for visually similar dogs
- occasional false dog detections
- overlapping barking events

Static intro/outro graphics are substantially reduced by the v3 static-scene filtering stage.

---

# Experimental Scripts

The following scripts are experimental and are **not** part of the final reported pipeline.

```
split_identity_switches_v2_5.py
```

Purpose

Research prototype for automatically detecting identity switches inside a single track before profile construction.

This script was investigated but is not included in the final evaluation results.
