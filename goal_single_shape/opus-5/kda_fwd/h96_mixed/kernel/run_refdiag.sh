set -x
export PIP_CONSTRAINT=
for i in 1 2 3; do pip install -q flashinfer-python && break; sleep 5; done
python -u -c "import flashinfer;print('fi',flashinfer.__version__)"
timeout 600 python -u refdiag.py 2>&1; echo "EXIT=$?"
