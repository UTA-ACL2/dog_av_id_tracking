#!/usr/bin/env bash
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
