#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
bash -n run_full_pipeline.sh
bash -n setup_environment.sh
python -m py_compile \
  01_visual/predict_active_barker.py \
  01_visual/score_bark_events.py \
  01_visual/all_dog_profile_manager_v2_4_1.py \
  02_audio/audio_fingerprint.py \
  03_fusion/fuse_visual_audio_v2.py
echo "Core syntax checks passed."
if [[ -f config.env ]]; then
  set -a
  source config.env
  set +a
  for key in DOG2VEC_MODEL FAIRSEQ_PATH RESNET_CHECKPOINT HYBRID_CHECKPOINT ACTIVE_BARKER_MODEL_DIR; do
    value="${!key:-}"
    if [[ -z "$value" ]]; then
      echo "MISSING: $key"
    elif [[ ! -e "$value" ]]; then
      echo "NOT FOUND: $key=$value"
    else
      echo "OK: $key"
    fi
  done
else
  echo "WARNING: config.env is missing."
fi

echo
echo "Remaining hardcoded path references:"
grep -RIn \
  --exclude-dir="__pycache__" \
  --exclude-dir="portability_backups" \
  --exclude="*.pyc" \
  --exclude="config.env" \
  --exclude="HARDCODED_PATH_AUDIT.txt" \
  -E "/home/|/Users/|~/multimodal_pipeline" \
  . || true
