set -e
NF="-gencode arch=compute_100a,code=sm_100a -O3 -std=c++17"
echo "=== [1] build v2 modes 0..3,9 ==="
for M in 0 1 2 3 9; do
  nvcc $NF -DFA_MODE=$M -Xptxas -v --shared -Xcompiler -fPIC fa_fwd2.cu -o libfa_fwd_m$M.so 2>&1 | grep -E "error|registers|spill|smem" | head -5
  ls -lh libfa_fwd_m$M.so
done
echo "=== [2] mode0: full (baseline v2-lazy) ==="
ln -sf libfa_fwd_m0.so libfa_fwd.so
python -u prof_dump.py 2>&1 | grep -v "UserWarning\|exe_args\|USDT"
echo "=== [3] mode1: no softmax math ==="
ln -sf libfa_fwd_m1.so libfa_fwd.so
python -u prof_dump.py 2>&1 | grep -v "UserWarning\|exe_args\|USDT"
echo "=== [4] mode2: no PV MMA ==="
ln -sf libfa_fwd_m2.so libfa_fwd.so
python -u prof_dump.py 2>&1 | grep -v "UserWarning\|exe_args\|USDT"
echo "=== [5] mode3: no QK+PV MMA ==="
ln -sf libfa_fwd_m3.so libfa_fwd.so
python -u prof_dump.py 2>&1 | grep -v "UserWarning\|exe_args\|USDT"
echo "ALL_MODES_DONE"
