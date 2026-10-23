#!/bin/bash
set -e
cd "$WORK_DIR"
pip install safetensors -q 2>/dev/null
python -u bench_ref.py
