#!/usr/bin/env bash
# Replace the paths below with your finalized scripts.

python 01_visual/predict_multimodal_barker_v2.py "$@"

python 01_visual/all_dog_profile_manager_v2_4_1.py "$@"

python 03_fusion/fuse_visual_audio_v2.py "$@"

python 04_evaluation/finalize_v2_4_1_evaluation.py "$@"
