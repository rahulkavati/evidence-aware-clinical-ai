#!/usr/bin/env bash
# Download the full PhysioNet/CinC Challenge 2019 training data (open access,
# about 40,336 patient files) into src/data/raw/physionet2019.
# Run from the repository root on your own machine:  bash scripts/download_physionet2019.sh
set -euo pipefail
DEST="src/data/raw/physionet2019"
BASE="https://physionet.org/files/challenge-2019/1.0.0/training"
mkdir -p "$DEST"
for SET in training_setA training_setB; do
  echo "Downloading $SET ..."
  wget -q -r -N -c -np -nH --cut-dirs=5 -A "*.psv" -P "$DEST/$SET" "$BASE/$SET/" \
    || { echo "wget failed; install it with 'brew install wget'"; exit 1; }
done
echo "Patient files: $(find "$DEST" -name '*.psv' | wc -l | tr -d ' ') (expected 40336)"
python scripts/check_data.py
