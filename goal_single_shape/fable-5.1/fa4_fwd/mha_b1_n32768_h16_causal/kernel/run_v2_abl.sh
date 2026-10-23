set -e
NF="-gencode arch=compute_100a,code=sm_100a -O3 -std=c++17"
echo "=== [1] build v2 mode 9 (profiling) ==="
nvcc $NF -DFA_MODE=9 -Xptxas -v --shared -Xcompiler -fPIC fa_fwd2.cu -o libfa_fwd.so 2>&1 | grep -E "error|registers|spill|smem" | head -5
ls -lh libfa_fwd.so
echo "=== [2] run prof_dump ==="
python -u prof_dump.py 2>&1 | grep -v "UserWarning\|exe_args\|USDT"
echo "PROF_DONE"
