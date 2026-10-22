export PIP_CONSTRAINT=
export NO_REF=1
for i in 1 2 3 4 5; do pip install -q --no-input "triton==3.5.0" && break; sleep 10; done
python -u -c "import triton;print('triton',triton.__version__)"
python -u final_run.py "$CFGS" 2>&1
