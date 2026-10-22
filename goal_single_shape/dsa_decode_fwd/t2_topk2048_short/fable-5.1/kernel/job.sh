#!/usr/bin/env bash
# Entry script executed inside the B200 batch job (cwd = unpacked bundle).
# DSA_VARIANTS: space-separated "name:nvcc -D flags" tuning builds benchmarked next to libdsa.so.
set -uo pipefail
trap 'echo "job.sh: EXIT status=$? line=$LINENO"' EXIT
trap 'echo "job.sh: got SIGTERM"' TERM
trap 'echo "job.sh: got SIGPIPE"' PIPE
[ -f job_env.sh ] && { echo "=== job_env.sh ==="; cat job_env.sh; source job_env.sh; }
# The log collector has silently dropped whole jobs after the CuTe-DSL JIT started: keep every line
# printable ASCII and short, and never let a broken stdout pipe kill the driver script.
sanitize() { stdbuf -oL tr -cd '\11\12\15\40-\176' | stdbuf -oL cut -c1-3000; }
run_py() { # run_py <label> <args...>: python with sanitized output, retried once on abnormal exit
  local label=$1; shift
  for attempt in 1 2; do
    python -u "$@" 2>&1 | sanitize
    local rc=${PIPESTATUS[0]}
    echo "$label exit=$rc (attempt $attempt)"
    [ "$rc" -lt 128 ] && return "$rc"   # >=128: killed by a signal -> retry
  done
  return 1
}
echo "=== env ==="
nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.max.sm,memory.total --format=csv
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cc', torch.cuda.get_device_capability())"
nvcc --version | tail -1

echo "=== pip deps for the reference (CuTe-DSL + TVM FFI launch path) ==="
export PIP_CONSTRAINT=
PKGS=("nvidia-cutlass-dsl==4.6.0" "cuda-python>=12.9,<13" "apache-tvm-ffi")
pip_ok=0
for idx in "" "-i https://pypi.tuna.tsinghua.edu.cn/simple" "-i https://mirrors.aliyun.com/pypi/simple/" ""; do
  if timeout 400 pip install -q --root-user-action=ignore --timeout 30 --retries 1 $idx "${PKGS[@]}" > /tmp/pip.log 2>&1; then
    echo "pip ok via '${idx:-pypi}'"; pip_ok=1; break
  fi
  echo "pip failed via '${idx:-pypi}': $(grep -v DEPRECATION /tmp/pip.log | tail -1)"
done
[ $pip_ok = 1 ] || echo "PIP_FAILED"
python -c "import cutlass, cuda.bindings.driver, tvm_ffi; print('cutlass dsl', cutlass.__version__, 'tvm_ffi', getattr(tvm_ffi, '__version__', '?'))" 2>&1 | tail -1

echo "=== reference repo (black box, never read): bundled blob or clone ==="
if [ -f ref_repo/kernels/dsa_attention/kernel.py ]; then
  echo "ref commit $(cat ref_repo/COMMIT) (bundled file, $(wc -c < ref_repo/kernels/dsa_attention/kernel.py) bytes)"
else
  for i in 1 2 3; do
    git clone -q --depth 1 https://github.com/mit-han-lab/mlsys2026-flashinfer-contest-solution.git ref_repo && break
    sleep 10
  done
  (cd ref_repo && echo "ref commit $(git rev-parse HEAD)")
fi

echo "=== GPU clocks (try to lock SM clock like the contest harness) ==="
nvidia-smi -lgc 1965,1965 2>&1 | tail -1 || true
nvidia-smi --query-gpu=clocks.sm,clocks.mem,clocks_throttle_reasons.active --format=csv

echo "=== build new kernel (+ variants) ==="
build() { # build <out.so> [flags...]
  local out=$1; shift
  nvcc -O3 -std=c++17 -arch=sm_100a -lineinfo -diag-suppress 177 -cudart shared -Xcompiler -fPIC -shared "$@" dsa_kernel.cu -o "$out" || { echo "NVCC_FAILED $out"; return 1; }
  echo "built $out flags: $*"
  cuobjdump -res-usage "$out" 2>/dev/null | grep -A1 dsa_sparse_attn | grep REG | sed 's/^/   /'
  cuobjdump -sass "$out" 2>/dev/null | python3 sass_summary.py "$out"
}
build libdsa.so ${DSA_MAIN_FLAGS:-} || exit 1
for spec in ${DSA_VARIANTS:-}; do
  name=${spec%%:*}; flags=${spec#*:}; flags=${flags//,/ }
  build "libdsa_${name}.so" $flags
done
if [ "${DSA_BUILD_MMA:-0}" = "1" ]; then
  for v in "mma:" "mmatiming:-DDSA_TIMING=1"; do
    name=${v%%:*}; flags=${v#*:}
    nvcc -O3 -std=c++17 -arch=sm_100a -lineinfo -diag-suppress 177 -cudart shared -Xcompiler -fPIC -shared $flags dsa_mma.cu -o libdsa_${name}.so \
      && echo "built libdsa_${name}.so" || echo "NVCC_FAILED libdsa_${name}.so"
    cuobjdump -res-usage libdsa_${name}.so 2>/dev/null | grep -A1 dsa_mma_kernel | grep REG | sed 's/^/   /'
    cuobjdump -sass libdsa_${name}.so 2>/dev/null | python3 sass_summary.py libdsa_${name}.so dsa_mma_kernel
  done
fi

echo "=== accuracy check (kernel under test) ==="
export CUDA_LAUNCH_BLOCKING=1   # checks only: surface device faults synchronously with a Python traceback
DSA_LIBS_ONLY=none run_py check check.py
if [ -n "${DSA_VARIANTS:-}" ]; then
  echo "=== accuracy check (variants, separate process) ==="
  names=$(for spec in $DSA_VARIANTS; do printf '%s,' "${spec%%:*}"; done)
  DSA_LIBS_ONLY="${names%,}" run_py check-variants check.py | grep -E 'variant|MISMATCH|RESULT\] accuracy|REFERENCE FAILED|exit='
fi
if [ -f libdsa_mma.so ]; then
  echo "=== accuracy check (mma kernel, separate process) ==="
  DSA_LIBS_ONLY=mma,mmatiming run_py check-mma check.py
fi

echo "=== benchmark ==="
unset CUDA_LAUNCH_BLOCKING
run_py bench bench.py
nvidia-smi -rgc > /dev/null 2>&1 || true
