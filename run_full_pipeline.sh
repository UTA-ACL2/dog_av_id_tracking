#!/usr/bin/env bash
set -euo pipefail

RELEASE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${CONFIG_FILE:-$RELEASE_ROOT/config.env}"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Missing configuration: $CONFIG_FILE"
  echo "Create it with:"
  echo "  cp \"$RELEASE_ROOT/config.example.env\" \"$RELEASE_ROOT/config.env\""
  exit 1
fi

set -a
source "$CONFIG_FILE"
set +a

CONDA_ENV_NAME="${CONDA_ENV_NAME:-dog2vec_gpu}"
DEVICE="${DEVICE:-cuda:0}"
AUDIO_THRESHOLD="${AUDIO_THRESHOLD:-0.60}"
PROFILE_VISUAL_THRESHOLD="${PROFILE_VISUAL_THRESHOLD:-0.65}"

usage() {
  cat <<'HELP'
Usage:
  ./run_full_pipeline.sh \
    --video-id VIDEO_ID \
    --video /path/to/video.mp4 \
    --timestamps /path/to/timestamps.csv \
    [--output-root /path/to/outputs] \
    [--device cuda:0] \
    [--audio-threshold 0.60] \
    [--profile-visual-threshold 0.65] \
    [--force]
HELP
}

VIDEO_ID=""
VIDEO=""
TIMESTAMPS=""
OUTPUT_ROOT_ARG=""
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --video-id) VIDEO_ID="$2"; shift 2 ;;
    --video) VIDEO="$2"; shift 2 ;;
    --timestamps) TIMESTAMPS="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT_ARG="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --audio-threshold) AUDIO_THRESHOLD="$2"; shift 2 ;;
    --profile-visual-threshold) PROFILE_VISUAL_THRESHOLD="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$VIDEO_ID" || -z "$VIDEO" || -z "$TIMESTAMPS" ]]; then
  usage
  exit 1
fi

OUTPUT_ROOT_ARG="${OUTPUT_ROOT_ARG:-${OUTPUT_ROOT:-}}"
if [[ -z "$OUTPUT_ROOT_ARG" ]]; then
  echo "Set --output-root or OUTPUT_ROOT in config.env"
  exit 1
fi

[[ -f "$VIDEO" ]] || { echo "Video not found: $VIDEO"; exit 1; }
[[ -f "$TIMESTAMPS" ]] || { echo "Timestamps not found: $TIMESTAMPS"; exit 1; }

required=(DOG2VEC_MODEL FAIRSEQ_PATH RESNET_CHECKPOINT HYBRID_CHECKPOINT ACTIVE_BARKER_MODEL_DIR)
for key in "${required[@]}"; do
  value="${!key:-}"
  [[ -n "$value" ]] || { echo "Missing config variable: $key"; exit 1; }
  [[ -e "$value" ]] || { echo "Configured path not found: $key=$value"; exit 1; }
done

BASE_DIR="$OUTPUT_ROOT_ARG/${VIDEO_ID}_base"
PROFILE_DIR="$OUTPUT_ROOT_ARG/${VIDEO_ID}_profiles"
FUSION_DIR="$OUTPUT_ROOT_ARG/${VIDEO_ID}_fusion"

PREDICT_ACTIVE="$RELEASE_ROOT/01_visual/predict_active_barker.py"
PROFILE_MANAGER="$RELEASE_ROOT/01_visual/all_dog_profile_manager_v2_4_1.py"
FUSION_SCRIPT="$RELEASE_ROOT/03_fusion/fuse_visual_audio_v2.py"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

export PYTHONPATH="$RELEASE_ROOT/01_visual:$RELEASE_ROOT/02_audio:$RELEASE_ROOT/03_fusion:${PYTHONPATH:-}"

mkdir -p "$OUTPUT_ROOT_ARG"
if [[ "$FORCE" -eq 1 ]]; then
  rm -rf "$BASE_DIR" "$PROFILE_DIR" "$FUSION_DIR"
fi

echo "[1/3] Tracking and active-barker localization"
python "$PREDICT_ACTIVE" \
  --video "$VIDEO" \
  --timestamps "$TIMESTAMPS" \
  --output-dir "$BASE_DIR" \
  --device "$DEVICE"

[[ -f "$BASE_DIR/tracks.csv" ]] || { echo "Missing tracks.csv"; exit 1; }
[[ -f "$BASE_DIR/full_audio.wav" ]] || { echo "Missing full_audio.wav"; exit 1; }

echo "[2/3] Persistent visual profiles"
python "$PROFILE_MANAGER" \
  --video "$VIDEO" \
  --tracks "$BASE_DIR/tracks.csv" \
  --output-dir "$PROFILE_DIR" \
  --device "$DEVICE" \
  --visual-similarity-threshold "$PROFILE_VISUAL_THRESHOLD" \
  --duplicate-iou-threshold 0.35 \
  --duplicate-appearance-threshold 0.75 \
  --part-duplicate-appearance-threshold 0.88 \
  --part-duplicate-min-iou 0.08 \
  --part-duplicate-min-intersection-over-smaller 0.25 \
  --part-duplicate-max-normalized-center-distance 0.50 \
  --intro-outro-window-sec 15

[[ -f "$PROFILE_DIR/track_to_all_dog_profile.csv" ]] || { echo "Missing track_to_all_dog_profile.csv"; exit 1; }
[[ -f "$PROFILE_DIR/all_dog_visual_profiles.json" ]] || { echo "Missing all_dog_visual_profiles.json"; exit 1; }

if [[ ! -s "$TIMESTAMPS" || ! -f "$BASE_DIR/predictions.csv" ]]; then
  echo "No valid bark events; fusion skipped."
  exit 0
fi

echo "[3/3] Audio-visual fusion"
mkdir -p "$FUSION_DIR"
python "$FUSION_SCRIPT" \
  --predictions "$BASE_DIR/predictions.csv" \
  --track-profile-map "$PROFILE_DIR/track_to_all_dog_profile.csv" \
  --visual-profiles "$PROFILE_DIR/all_dog_visual_profiles.json" \
  --wav "$BASE_DIR/full_audio.wav" \
  --output-dir "$FUSION_DIR" \
  --dog2vec-model "$DOG2VEC_MODEL" \
  --fairseq-path "$FAIRSEQ_PATH" \
  --resnet-checkpoint "$RESNET_CHECKPOINT" \
  --hybrid-checkpoint "$HYBRID_CHECKPOINT" \
  --device "$DEVICE" \
  --audio-threshold "$AUDIO_THRESHOLD" \
  --save-bark-clips

[[ -f "$FUSION_DIR/pipeline_summary_v2.csv" ]] || { echo "Missing pipeline_summary_v2.csv"; exit 1; }

echo
echo "Pipeline complete"
column -s, -t < "$FUSION_DIR/pipeline_summary_v2.csv" || cat "$FUSION_DIR/pipeline_summary_v2.csv"
