#!/bin/bash
export PIP_CONSTRAINT=
export PIP_DISABLE_PIP_VERSION_CHECK=1
inst() { timeout 420 pip install -q --no-input "$@" 2>&1 | grep -viE "DEPRECATION|WARNING: Running pip" | tail -1; }
inst apache-tvm-ffi
inst --no-deps nvidia-cudnn-frontend
inst --no-deps einops ninja requests filelock packaging tqdm click tabulate cuda-python nvidia-ml-py
inst fla-core
inst --no-deps flashinfer-python==0.6.18.post1
inst -U nvidia-cutlass-dsl
python -c "import importlib.metadata as m; print('flashinfer', m.version('flashinfer-python'), '| cutlass-dsl', m.version('nvidia-cutlass-dsl'), '| fla', m.version('fla-core'))"
echo "########## FINAL"; timeout 2400 python -u final.py 2>&1
echo "########## DONE"
