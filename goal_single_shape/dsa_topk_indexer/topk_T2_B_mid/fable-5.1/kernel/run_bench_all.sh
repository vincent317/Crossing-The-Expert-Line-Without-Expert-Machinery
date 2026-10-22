#!/bin/bash
set -e
cd "$WORK_DIR"
export PIP_CONSTRAINT=
pip install safetensors -q 2>&1 | grep -v "already satisfied" || true
python -u bench_all.py
