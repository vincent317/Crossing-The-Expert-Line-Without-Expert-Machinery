set -x
nvcc --version
python -c "import torch;print('torch',torch.__version__, torch.version.cuda)"
python -c "import cutlass, os; print('cutlass pkg', cutlass.__file__)" 2>&1 | head -3
python - <<'PY'
import glob, os
cands = []
for base in ("/opt/sglang/lib/python3.12/site-packages", "/usr/local/lib/python3.12/site-packages", "/usr/local", "/opt"):
    for pat in ("**/cute/atom/mma_traits_sm100*.hpp", "**/cutlass/arch/mma_sm100*.hpp", "**/include/cutlass/cutlass.h"):
        cands += glob.glob(os.path.join(base, pat), recursive=True)[:5]
print("CUTLASS_HDRS:")
for c in sorted(set(cands)): print(" ", c)
PY
pip download nvidia-cutlass-dsl==0 2>&1 | head -2
ls /usr/local/cuda/include/cuda_fp8.h && echo FP8_OK
nvcc -arch=sm_100a -c probe_tcgen05.cu -o /tmp/pt.o 2>&1 | head -40 && echo TCGEN05_ASM_OK
nvcc -arch=sm_100a -O3 probe.cu -o /tmp/probe 2>&1|head -20 && /tmp/probe
