set -x
export PIP_CONSTRAINT=
CFGS="16:10:128:8:2:0,16:10:64:4:2:0,16:10:64:4:3:0,16:10:64:8:2:0,32:5:64:4:2:0,32:5:128:8:2:0,32:5:64:8:2:0,16:10:32:4:2:0,32:5:32:4:2:0,16:10:64:4:2:1"
for i in 1 2 3 4 5 6; do pip install -q triton==3.3.1 && break; sleep 5; done
python -u -c "import triton;print('=== TRITON', triton.__version__)"
timeout 1200 python -u check_mine.py "$CFGS" 2>&1
echo "PHASE1_EXIT=$?"
for i in 1 2 3 4 5 6; do pip install -q triton==3.4.0 && break; sleep 5; done
python -u -c "import triton;print('=== TRITON', triton.__version__)"
timeout 1200 python -u check_mine.py "$CFGS" 2>&1
echo "PHASE2_EXIT=$?"
