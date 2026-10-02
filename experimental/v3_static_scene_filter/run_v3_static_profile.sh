#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 VIDEO_ID VIDEO_PATH TRACKS_CSV"
  exit 1
fi

VIDEO_ID="$1"
VIDEO_PATH="$2"
TRACKS_CSV="$3"

ROOT='/path/to/outputs'
FILTER_OUT="${ROOT}/v3_static_filter_${VIDEO_ID}"
PROFILE_OUT="${ROOT}/v3_profiles_${VIDEO_ID}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dog2vec_gpu

rm -rf "${FILTER_OUT}" "${PROFILE_OUT}"

python "${ROOT}/filter_static_dog_graphics_v3.py"   --video "${VIDEO_PATH}"   --tracks "${TRACKS_CSV}"   --output-dir "${FILTER_OUT}"   --boundary-window-sec 20   --analysis-step-sec 0.5   --static-motion-threshold 3.0   --min-static-duration-sec 3.0   --min-simultaneous-dog-tracks 2   --track-static-fraction-threshold 0.80

python "${ROOT}/all_dog_profile_manager_v2_4_1.py"   --video "${VIDEO_PATH}"   --tracks "${FILTER_OUT}/tracks_live_scene_v3.csv"   --output-dir "${PROFILE_OUT}"   --device cuda:0   --visual-similarity-threshold 0.65   --duplicate-iou-threshold 0.35   --duplicate-appearance-threshold 0.75   --part-duplicate-appearance-threshold 0.88   --part-duplicate-min-iou 0.08   --part-duplicate-min-intersection-over-smaller 0.25   --part-duplicate-max-normalized-center-distance 0.50   --intro-outro-window-sec 15

echo
echo "Filter output:  ${FILTER_OUT}"
echo "Profile output: ${PROFILE_OUT}"
