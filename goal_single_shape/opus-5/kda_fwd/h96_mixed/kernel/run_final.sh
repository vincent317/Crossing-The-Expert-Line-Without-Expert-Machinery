set -x
export PIP_CONSTRAINT=
for i in 1 2 3; do pip install -q flashinfer-python && break; sleep 5; done
for i in 1 2 3 4 5 6 7 8; do pip install -q triton==3.4.0 && break; sleep 5; done
python -u -c "import triton;print('triton',triton.__version__)"
timeout 2400 python -u final.py "$CFGS" 2>&1
echo "EXIT=$?"
