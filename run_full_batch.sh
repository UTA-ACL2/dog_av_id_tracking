#!/usr/bin/env bash
set -u

OUTPUT_ROOT=/path/to/outputs/full_pipeline_runs
VIDEO_ROOT=/path/to/videos
ANNOT_ROOT=/path/to/annotations
TIMESTAMP_ROOT=/path/to/outputs/inputs_timestamps

mkdir -p "$OUTPUT_ROOT"

for VIDEO_DIR in "$ANNOT_ROOT"/*; do
  [[ -d "$VIDEO_DIR" ]] || continue

  VIDEO_ID=$(basename "$VIDEO_DIR")

  VIDEO=$(find "$VIDEO_ROOT" \
    -maxdepth 1 \
    -type f \
    -name "${VIDEO_ID}_*.mp4" \
    | head -n 1)

  TIMESTAMPS="$TIMESTAMP_ROOT/$VIDEO_ID/timestamps.csv"

  if [[ -z "$VIDEO" ]]; then
    echo "SKIP $VIDEO_ID: original video not found"
    continue
  fi

  if [[ ! -f "$TIMESTAMPS" ]]; then
    echo "SKIP $VIDEO_ID: timestamps not found"
    continue
  fi

  echo
  echo "============================================================"
  echo "RUNNING VIDEO $VIDEO_ID"
  echo "VIDEO: $VIDEO"
  echo "============================================================"

  if ./run_full_pipeline.sh \
    --video-id "$VIDEO_ID" \
    --video "$VIDEO" \
    --timestamps "$TIMESTAMPS" \
    --output-root "$OUTPUT_ROOT" \
    --force
  then
    echo "COMPLETE $VIDEO_ID"
  else
    echo "FAILED $VIDEO_ID"
  fi
done
