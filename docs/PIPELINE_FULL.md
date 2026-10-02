# Detailed Pipeline Documentation

## 1. Purpose

This document describes the complete final multimodal dog fingerprinting pipeline and the design decisions that produced version 2.4.1.

The task is to process an unconstrained video containing one or more dogs and produce persistent dog identities that remain meaningful throughout the video. The pipeline must support:

- several dogs visible at once;
- dogs that look very similar;
- occlusion;
- dogs leaving and re-entering the camera view;
- silent dogs;
- off-screen barking;
- tracker fragmentation;
- duplicate detections;
- head/body detections of the same dog;
- intro and outro dog images;
- disagreement between visual and audio evidence.

The final output is a set of persistent multimodal profiles, not a set of raw tracker IDs.

---

# 2. Terminology

## Temporary track

A short-term ID created by the multi-object tracker.

Example:

```text
track 0
track 1
track 2
```

Temporary tracks are observations, not guaranteed physical identities.

## Persistent visual profile

A long-term identity created by merging compatible temporary tracks.

Example:

```text
track 0 + track 4 + track 7 → dog_1
```

## Audio profile

An accumulated audio representation created from bark embeddings assigned to a dog.

## Final multimodal profile

A persistent identity containing visual tracks, bark events, audio evidence, and status information.

---

# 3. End-to-End Architecture

```text
Input video
    │
    ├── Extract audio
    │
    ├── Detect dogs
    │
    └── Track detections
             ↓
      Temporary tracks
             ↓
      Track quality analysis
             ↓
      OSNet appearance encoding
             ↓
      Pairwise track diagnostics
             ↓
      Duplicate and fragment merging
             ↓
      Persistent visual profiles
             ↓
 Bark intervals + active-barker prediction
             ↓
 Track-to-profile mapping
             ↓
      Audio fingerprint extraction
             ↓
      Audio-profile comparison
             ↓
      Visual-audio fusion
             ↓
 Final persistent multimodal dog profiles
```

---

# 4. Inputs

## Video

Expected format:

```text
.mp4
```

The video path is used by:

- dog detection;
- tracking;
- crop extraction;
- OSNet appearance encoding;
- internal-motion analysis;
- audio extraction.

## Bark intervals

Expected CSV:

```text
event_id,start_time,end_time,start_frame,end_frame
```

Example:

```text
gt_bark_000,21.788,23.457,653,702
```

Bark intervals may come from a detector or annotation file. For the finalized evaluation, the intervals were derived from our frame-level `Vocalization=Barking` annotations.

A video may contain no bark intervals. Such videos remain valid for visual-profile evaluation.

---

# 5. Dog Detection

The detector generates a bounding box and confidence score for each visible dog.

A detection is represented by:

```text
frame
time
x1
y1
x2
y2
confidence
```

The detector is not expected to be perfect. Common errors include:

- missed dogs;
- duplicate boxes;
- boxes covering only a head or rear body region;
- one box covering two overlapping dogs;
- dog images or logos detected as live dogs.

These errors motivate the downstream profile manager.

---

# 6. Temporary Multi-Object Tracking

The tracker associates detections across time and assigns `track_id`.

The tracker is useful for short-term continuity, but it is not treated as the final identity layer.

Observed failure modes:

## Fragmentation

```text
Dog A → track 0
occlusion
Dog A → track 5
```

## Identity switch

```text
track 1 starts on Dog B
track 1 later follows Dog C
```

## Duplicate simultaneous tracks

```text
track 2 → dog head
track 3 → dog body
```

## Merged detection

```text
one box covers Dog A and Dog B
```

## False visual profile

```text
outro dog image → tracker ID
```

The profile manager repairs fragmentation and several duplicate cases, but it does not fully solve within-track identity switches or merged multi-dog boxes.

---

# 7. Track Statistics

For every temporary track, the system computes:

```text
first_frame
last_frame
first_time_sec
last_time_sec
track_rows
frame_span
observation_density
mean_confidence
box width and height
box diagonal
total center movement
normalized movement
```

Observation density is:

```text
number of observed rows / first-to-last frame span
```

This is important because a track may be sparse even when its first-to-last span is large.

The original implementation incorrectly treated every frame between a track's first and last frame as an observation. v2.1 corrected this by using actual observed-frame sets.

---

# 8. Appearance Embeddings

The visual profile manager samples up to 12 crops from each temporary track.

For each sampled row:

1. Read the source video frame.
2. Crop the bounding box.
3. Convert BGR to RGB.
4. Resize to the OSNet input size.
5. Normalize the image.
6. Run the pretrained OSNet encoder.
7. L2-normalize the output embedding.

The track centroid is the normalized mean of its valid sampled embeddings.

The OSNet model is generic and not trained specifically for dog identity. Therefore, it is useful for broad appearance consistency but may assign high similarity to different dogs with similar coats and breeds.

---

# 9. Internal Crop Motion

Bounding-box movement alone cannot distinguish a real stationary dog from a static image.

The pipeline therefore samples crops over time and computes grayscale pixel differences.

For each track:

1. Sample up to 16 crop observations.
2. Convert each crop to grayscale.
3. Resize crops to a common size.
4. Compute the mean absolute difference between consecutive crops.
5. Store mean, median, and maximum internal motion.

Interpretation:

```text
Low box motion + nonzero internal motion
→ possible stationary live dog

Low box motion + nearly zero internal motion
→ possible static image, screen, or graphic
```

This remains a heuristic. A completely still real dog can resemble a static image.

---

# 10. Track Classification

Tracks are assigned diagnostic classifications.

## `active_real_candidate`

The track has sufficient observations and meaningful visual change.

## `sparse_candidate`

The track has few observations relative to its lifespan.

Frozen sparse parameters:

```text
maximum observation density: 0.02
maximum rows:                20
```

A sparse track is not automatically discarded when it can be merged into a stronger profile.

## `static_uncertain`

The track has extremely low normalized box motion and very low internal crop motion, but it is not restricted to an intro or outro boundary.

## `likely_intro_outro_image`

The track:

- appears only near the start or end of the video;
- has very low box motion;
- has very low internal motion.

Frozen boundary window:

```text
15 seconds
```

By default, likely image tracks can remain marked as uncertain. Strict suppression is available but was used cautiously.

---

# 11. Pairwise Track Analysis

For every track pair, the profile manager computes:

```text
appearance similarity
common observed frames
overlap fraction
median IoU
mean IoU
median normalized edge gap
median normalized center distance
median intersection over smaller area
median area ratio
nearest frame gap
nearest time gap
```

## Appearance similarity

Cosine similarity between normalized track centroids.

## Common observed frames

The number of actual frames containing both tracks.

## IoU

Bounding-box intersection over union on common frames.

## Intersection over smaller

Intersection area divided by the smaller box area. This is useful when a head box partially overlaps a full-body box.

## Normalized center distance

Distance between box centers divided by their average diagonal scale.

## Nearest time gap

The smallest temporal separation between actual observations from two tracks.

All metrics are written to:

```text
visual_track_pair_diagnostics.csv
```

---

# 12. Merge Categories

## 12.1 Fragment Re-identification

Decision label:

```text
candidate_fragment_reid
```

Purpose:

Reconnect non-simultaneous track fragments belonging to the same dog.

Frozen threshold:

```text
visual similarity threshold: 0.65
maximum re-identification gap: 120 seconds
```

The pair must not contain strong evidence that two different dogs coexist.

This was the most common accepted merge type in the final evaluation.

Final count:

```text
38 merges
79.17% of accepted merges
```

---

## 12.2 Ordinary Duplicate Overlap

Decision label:

```text
candidate_duplicate_overlap
```

Purpose:

Merge two simultaneous tracks that strongly overlap the same dog.

Frozen parameters:

```text
duplicate IoU threshold:          0.35
duplicate appearance threshold:   0.75
minimum common frames:            2
```

Final count:

```text
2 merges
4.17% of accepted merges
```

---

## 12.3 Part-Based Duplicate Merge

Decision label:

```text
candidate_part_based_duplicate
```

Purpose:

Merge simultaneous detections that cover different parts of one dog.

Example discovered in video `50816`:

```text
track 2 → head/front region
track 3 → rear/body region
```

The initial v2.4 rule was too permissive and incorrectly merged nearby different dogs. v2.4.1 tightened the conditions.

Frozen v2.4.1 parameters:

```text
appearance similarity:                 at least 0.88
median IoU:                            at least 0.08
or intersection over smaller area:     at least 0.25
normalized center distance:            at most 0.50
```

Final count:

```text
8 merges
16.67% of accepted merges
```

---

# 13. Different-Dog Protection

The profile manager prevents appearance-only over-merging.

This is necessary because visually similar dogs can have OSNet cosine similarities above 0.90.

Tracks that coexist over several frames while occupying different spatial locations are treated as different dogs even when their appearance similarity is high.

Frozen protection parameters:

```text
different-dog IoU threshold:    0.20
substantial overlap frames:      3
```

The part-based duplicate rule is the only controlled exception, and it requires strict spatial evidence.

---

# 14. Persistent Visual Profile Creation

Accepted merges are processed with connected-component/union logic.

Each final component becomes:

```text
dog_1
dog_2
dog_3
...
```

Each profile contains:

```text
persistent profile ID
member track IDs
first seen time
last seen time
number of fragments
visual centroid
visual classifications
uncertainty flag
bark history
bark count
status
```

Mapping output:

```text
track_to_all_dog_profile.csv
```

Example:

```text
track 2 → dog_3
track 3 → dog_3
track 4 → dog_3
```

---

# 15. Bark-Event Scoring

Each bark interval is aligned with visible tracked dogs.

For every event, the active-barker scorer creates candidate scores and writes:

```text
event_id
start_time_sec
end_time_sec
predicted_track_id
top_score
second_score
margin
all_candidate_ids
all_scores
status
```

The selected temporary track is mapped to its persistent visual profile before fusion.

---

# 16. Audio Feature Extraction

Each bark clip is extracted from `full_audio.wav`.

The final hybrid audio model combines:

## Dog2Vec

```text
768-dimensional representation
```

## MFCC

```text
80-dimensional representation
```

## Log-Mel ResNet

```text
256-dimensional representation
```

Concatenated input:

```text
1104 dimensions
```

Fusion network:

```text
1104 → 1024 → 512 → 256
```

The final 256-dimensional fingerprint is L2-normalized.

Training-only ArcFace head:

```text
scale s = 30
margin m = 0.5
```

The evaluation pipeline uses cosine similarity between fingerprint embeddings and profile centroids.

Frozen online matching threshold:

```text
audio similarity threshold = 0.60
```

---

# 17. Audio Profile Management

For each new bark:

1. Extract the 256-dimensional embedding.
2. Compare it with existing audio profile centroids.
3. Select the highest-scoring profile.
4. Record the similarity.
5. Compare the audio suggestion with the visible visual profile.
6. Update the chosen profile when appropriate.

The online profile order can affect early assignments because the first valid bark initializes an audio centroid.

Pairwise audio analysis showed that short bark clips may not form clean identity clusters. Therefore, audio is not treated as an unquestionable source of identity.

---

# 18. Fusion Decision Policy

## Visible profile available

The bark is attached to the mapped visual profile.

Possible decision:

```text
attached_to_visible_all_dog_profile
```

## Audio agrees

Record agreement and update audio evidence.

## Audio disagrees

Keep the visible profile as the final assignment and record:

```text
audio_visual_identity_conflict
```

## No visible profile

Use audio to match an existing identity.

## No suitable audio match

Create an off-screen audio-only profile where supported.

This policy prevents unreliable short-clip audio similarities from overriding strong visual localization.

---

# 19. Final Per-Video Outputs

## `bark_to_profile_v2.csv`

Important fields:

```text
event_id
start_time_sec
end_time_sec
predicted_track_id
all_dog_profile_id
assignment_source
audio_best_profile_id
audio_similarity
audio_visual_agreement
v2_decision
audio_clip_status
audio_clip_path
```

## `all_dog_profiles_v2.json`

Persistent multimodal profiles and their histories.

## `pipeline_summary_v2.csv`

Summary fields:

```text
total_persistent_dogs
barking_dogs
silent_dogs
visual_profiles
offscreen_audio_only_profiles
bark_events
assigned_bark_events
audio_visual_conflicts
```

---

# 20. Final Evaluation Protocol

The final evaluator discovers every prepared video under:

```text
inputs/*/timestamps.csv
```

A video is considered visually complete when it has:

```text
tracks.csv
track_to_all_dog_profile.csv
```

A video is considered multimodally complete when it also has:

```text
pipeline_summary_v2.csv
bark_to_profile_v2.csv
```

An empty timestamp file is labeled:

```text
visual_complete_no_bark_events
```

It is not counted as a failure.

---

# 21. Final Evaluation Results

## Dataset coverage

```text
Videos discovered:                 23
Videos visually complete:          23
Videos multimodally complete:      22
Visual-only videos with no barks:   1
Missing visual outputs:             0
Missing fusion outputs:             0
```

## Visual identity consolidation

```text
Temporary tracks:                  148
Persistent visual profiles:       100
Fragments reduced:                 48
Overall profile reduction:      32.43%
Mean temporary tracks/video:      6.43
Mean persistent profiles/video:   4.35
Mean reduction/video:             2.09
```

## Merge breakdown

```text
Fragment re-identification:        38
Part-based duplicates:              8
Overlap duplicates:                 2
```

## Bark assignment

```text
Bark events:                       109
Assigned bark events:             108
Assignment coverage:            99.08%
Audio-visual conflicts:            14
Conflict rate:                  12.96%
```

---

# 22. Qualitative Review

Five high-value videos were reviewed manually:

```text
3921
42956
51669
79755
80461
```

Observations:

- `80461` showed strong tracking and no unnecessary merging.
- `79755` showed good identity consolidation.
- `3921`, `42956`, and `51669` remained difficult due to occlusion, similar-looking dogs, identity switches, and extra profiles.
- The most difficult cases involved dogs jumping on each other or crossing while visually similar.
- Some tracks switched identities.
- Some boxes contained two dogs simultaneously.

These failures are attributed primarily to the detector/tracker stage rather than simple profile-merging thresholds.

---

# 23. Known Limitations

## Within-track identity switches

The profile manager assumes a temporary track is internally consistent. A track that switches physical dogs can contaminate a final profile.

## Multi-dog bounding boxes

One box containing two dogs creates a mixed visual embedding and ambiguous identity.

## Similar-dog ReID

Generic OSNet features are not sufficiently discriminative for many same-breed dogs.

## Active-barker uncertainty

The visual scorer may select the wrong dog during subtle or overlapping barking.

## Audio confusion

Different dogs may have high audio similarity, while two barks from one dog may have lower similarity.

## Intro/outro heuristics

A fixed 15-second boundary window does not fit every video.

## No full physical-identity ground truth

The final 32.43% profile reduction cannot be interpreted as identity accuracy.

---

# 24. Experimental v2.5 Identity-Switch Preprocessor

Script:

```text
split_identity_switches_v2_5.py
```

Method:

- divides tracks into temporal windows;
- extracts window-level OSNet embeddings;
- looks for sustained appearance changes;
- checks spatial jumps;
- checks for nearby competing tracks;
- proposes split points.

Initial test on video `42956` produced several candidate switches, but some may correspond to pose changes, crop changes, or occlusion. Therefore, v2.5 remains experimental and is not included in the frozen results.

---

# 25. Recommended Future Improvements

## Dog-specific re-identification

Train visual embeddings specifically for individual dogs.

## Instance segmentation

Use masks rather than only boxes to separate overlapping dogs.

## Joint detection and tracking improvement

Reduce boxes that cover multiple dogs.

## Within-track segmentation

Split identity-switched tracks using globally validated evidence.

## Global graph optimization

Optimize all fragment assignments jointly instead of using only pairwise greedy merges.

## Reliability-aware fusion

Weight audio and visual evidence using confidence and margins.

## Better audio segmentation

Improve bark boundaries and handle overlapping vocalizations.

---

# 26. Reproducibility Checklist

Before running:

```text
[ ] Activate dog2vec_gpu
[ ] Confirm GPU availability
[ ] Confirm source video exists
[ ] Confirm timestamps.csv exists
[ ] Confirm model checkpoints exist
[ ] Confirm output directories are separate
```

After base pipeline:

```text
[ ] tracks.csv exists
[ ] predictions.csv exists for bark videos
[ ] full_audio.wav exists
[ ] logs contain no fatal error
```

After profile manager:

```text
[ ] track_to_all_dog_profile.csv exists
[ ] visual_track_merges.csv exists
[ ] visual_track_pair_diagnostics.csv exists
[ ] all_dog_visual_profiles.json exists
```

After fusion:

```text
[ ] bark_to_profile_v2.csv exists
[ ] all_dog_profiles_v2.json exists
[ ] pipeline_summary_v2.csv exists
```

After final evaluation:

```text
[ ] per_video_results.csv exists
[ ] aggregate_results.csv exists
[ ] merge_breakdown.csv exists
[ ] audio_visual_conflicts.csv exists
[ ] final_results.xlsx exists
[ ] paper_ready_results.txt exists
```

---

# 27. Frozen Version Statement

The final paper evaluation uses:

```text
all_dog_profile_manager_v2_4_1.py
fuse_visual_audio_v2.py
finalize_v2_4_1_evaluation.py
```

with the thresholds documented in this file.

Later experimental scripts must be evaluated separately and should not overwrite the frozen v2.4.1 result folders.
