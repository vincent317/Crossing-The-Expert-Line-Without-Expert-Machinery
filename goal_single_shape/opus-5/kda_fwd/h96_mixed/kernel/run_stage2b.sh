set -x
export PIP_CONSTRAINT=
pip index versions flashinfer-cubin 2>&1 | head -3
pip index versions flashinfer-jit-cache 2>&1 | head -3
for i in 1 2 3; do pip install -q flashinfer-python && break; sleep 5; done
pip install -q flashinfer-cubin 2>&1 | tail -3 || echo "no cubin pkg"
timeout 900 pip install -q flashinfer-jit-cache 2>&1 | tail -3 || echo "no jit-cache pkg"
python -u -c "import flashinfer;print('fi',flashinfer.__version__)"
timeout 2400 python -u stage2b.py 2>&1
echo "EXIT=$?"
