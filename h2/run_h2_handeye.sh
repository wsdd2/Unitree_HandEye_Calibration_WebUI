#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
python3 capture_h2_handeye.py "$@"
