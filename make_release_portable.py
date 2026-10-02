#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAMP = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
BACKUP = ROOT / 'portability_backups' / STAMP

CONFIG_EXAMPLE = '''# Portable configuration
CONDA_ENV_NAME=dog2vec_gpu

DOG2VEC_MODEL=/path/to/dog2vec_130k_9.pt
FAIRSEQ_PATH=/path/to/fairseq
RESNET_CHECKPOINT=/path/to/voxblink_logmel_resnet_best.pt
HYBRID_CHECKPOINT=/path/to/hybrid_fusion_best_by_eval.pt

ACTIVE_BARKER_MODEL_DIR=/path/to/models_final
RAW_VIDEOS_JSON=/path/to/raw_videos.json
SENTENCES_JSON=/path/to/sentences.json

DEVICE=cuda:0
AUDIO_THRESHOLD=0.60
PROFILE_VISUAL_THRESHOLD=0.65

VIDEOS_ROOT=/path/to/videos
ANNOTATIONS_ROOT=/path/to/vids_and_annotations
TIMESTAMP_INPUT_ROOT=/path/to/timestamp_inputs
OUTPUT_ROOT=/path/to/pipeline_outputs
'''

RUNNER = r'''#!/usr/bin/env bash
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
'''

SETUP = r'''#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$ROOT/environment.yml"
command -v conda >/dev/null 2>&1 || { echo "Conda not found."; exit 1; }
[[ -f "$ENV_FILE" ]] || { echo "Missing $ENV_FILE"; exit 1; }
source "$(conda info --base)/etc/profile.d/conda.sh"
ENV_NAME="$(awk -F': *' '/^name:/ {print $2; exit}' "$ENV_FILE")"
ENV_NAME="${ENV_NAME:-dog2vec_gpu}"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
else
  conda env create -f "$ENV_FILE"
fi
echo "Environment ready: $ENV_NAME"
'''

CHECK = r'''#!/usr/bin/env bash
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
  -E "/path/to/|/Users/|~/multimodal_pipeline" \
  . || true
'''

CHECKPOINT_README = '''Model files are configured through config.env.

Required variables:
DOG2VEC_MODEL
FAIRSEQ_PATH
RESNET_CHECKPOINT
HYBRID_CHECKPOINT
ACTIVE_BARKER_MODEL_DIR
'''


def backup(path: Path) -> None:
    if not path.exists():
        return
    target = BACKUP / path.relative_to(ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)


def write(path: Path, text: str, executable: bool = False) -> None:
    backup(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    if executable:
        path.chmod(path.stat().st_mode | 0o111)
    print('Wrote:', path.relative_to(ROOT))


def ensure_import_os(text: str) -> str:
    if re.search(r'^\s*import os\s*$', text, re.MULTILINE):
        return text
    future = re.search(r'^from __future__ import .+\n', text, re.MULTILINE)
    if future:
        i = future.end()
        return text[:i] + '\nimport os\n' + text[i:]
    shebang = re.match(r'^#!.*\n', text)
    i = shebang.end() if shebang else 0
    return text[:i] + 'import os\n' + text[i:]


def patch_core(path: Path, replacements: dict[str, str]) -> None:
    if not path.is_file():
        print('Missing, skipped:', path.relative_to(ROOT))
        return
    original = path.read_text(encoding='utf-8')
    updated = original
    changed = False
    for old, new in replacements.items():
        if old in updated:
            updated = updated.replace(old, new)
            changed = True
        else:
            print('Not found:', old, 'in', path.relative_to(ROOT))
    if changed:
        backup(path)
        path.write_text(ensure_import_os(updated), encoding='utf-8')
        print('Patched:', path.relative_to(ROOT))


def append_missing_config(path: Path) -> None:
    if not path.exists():
        write(path, CONFIG_EXAMPLE)
        return
    text = path.read_text(encoding='utf-8')
    keys = {line.split('=', 1)[0].strip() for line in text.splitlines() if '=' in line and not line.lstrip().startswith('#')}
    defaults = {
        'ACTIVE_BARKER_MODEL_DIR': '/path/to/models_final',
        'RAW_VIDEOS_JSON': '/path/to/raw_videos.json',
        'SENTENCES_JSON': '/path/to/sentences.json',
    }
    missing = [f'{k}={v}' for k, v in defaults.items() if k not in keys]
    if missing:
        backup(path)
        with path.open('a', encoding='utf-8') as f:
            f.write('\n# Added by make_release_portable.py\n')
            f.write('\n'.join(missing) + '\n')
        print('Updated:', path.relative_to(ROOT))


def main() -> None:
    if not (ROOT / '01_visual').is_dir():
        raise SystemExit('Place this script in FINAL_MODEL_RELEASE and run it there.')

    BACKUP.mkdir(parents=True, exist_ok=True)
    print('Backups:', BACKUP)

    write(ROOT / 'config.example.env', CONFIG_EXAMPLE)
    append_missing_config(ROOT / 'config.env')
    write(ROOT / 'run_full_pipeline.sh', RUNNER, True)
    write(ROOT / 'setup_environment.sh', SETUP, True)
    write(ROOT / 'check_portability.sh', CHECK, True)
    write(ROOT / 'checkpoints' / 'README.txt', CHECKPOINT_README)

    patch_core(ROOT / '01_visual' / 'predict_active_barker.py', {
        '"/path/to/models_final/"': 'os.environ.get("ACTIVE_BARKER_MODEL_DIR", "/path/to/models_final/")',
        '"/path/to/raw_videos.json"': 'os.environ.get("RAW_VIDEOS_JSON", "/path/to/raw_videos.json")',
        '"/path/to/sentences.json"': 'os.environ.get("SENTENCES_JSON", "/path/to/sentences.json")',
    })

    patch_core(ROOT / '01_visual' / 'score_bark_events.py', {
        '"/path/to/models_final/"': 'os.environ.get("ACTIVE_BARKER_MODEL_DIR", "/path/to/models_final/")',
    })

    gitignore = ROOT / '.gitignore'
    lines = gitignore.read_text(encoding='utf-8').splitlines() if gitignore.exists() else []
    for item in ['config.env', '__pycache__/', '*.pyc', '*.pyo', '.DS_Store', 'outputs/', 'logs/', 'portability_backups/', 'HARDCODED_PATH_AUDIT.txt']:
        if item not in lines:
            lines.append(item)
    write(gitignore, '\n'.join(lines) + '\n')

    audit = subprocess.run([
        'grep', '-RIn',
        '--exclude-dir=__pycache__',
        '--exclude-dir=portability_backups',
        '--exclude=*.pyc',
        '--exclude=config.env',
        '--exclude=HARDCODED_PATH_AUDIT.txt',
        '-E', r'/home/|/Users/|~/multimodal_pipeline',
        '.',
    ], cwd=ROOT, text=True, capture_output=True)
    (ROOT / 'HARDCODED_PATH_AUDIT.txt').write_text(audit.stdout, encoding='utf-8')

    checks = [
        (['bash', '-n', 'run_full_pipeline.sh'], 'runner'),
        (['bash', '-n', 'setup_environment.sh'], 'environment setup'),
        ([sys.executable, '-m', 'py_compile',
          '01_visual/predict_active_barker.py',
          '01_visual/score_bark_events.py',
          '01_visual/all_dog_profile_manager_v2_4_1.py',
          '02_audio/audio_fingerprint.py',
          '03_fusion/fuse_visual_audio_v2.py'], 'core Python'),
    ]

    failed = False
    for command, label in checks:
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode:
            failed = True
            print('FAILED:', label)
        else:
            print('OK:', label)

    print('\nDone.')
    print('1. Fill the new variables in config.env.')
    print('2. Run ./check_portability.sh')
    print('3. Test one known video.')
    print('Backups:', BACKUP)

    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
