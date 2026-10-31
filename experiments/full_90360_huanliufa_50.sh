#!/usr/bin/env bash
set -euo pipefail
python full_data_nebula/submit.py --config experiments/full_90360_huanliufa_50.json "$@"
