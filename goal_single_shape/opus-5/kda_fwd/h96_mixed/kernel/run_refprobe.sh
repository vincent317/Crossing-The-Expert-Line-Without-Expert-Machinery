set -x
export PIP_CONSTRAINT=
for i in 1 2 3; do pip install -q flashinfer-python && break; sleep 5; done
timeout 2700 python -u refprobe.py 2>&1; echo "EXIT=$?"
