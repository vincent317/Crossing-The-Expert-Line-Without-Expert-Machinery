set -e
R=/jiuzhang_train/user/chenyaojian
W="$R/kernelgoal-remainder-03413a0/topk-baseline-red-r3"
mkdir -p "$W"
cd "$W"
mkdir -p "$W/tmp" "$W/cache/cuda"
export TMPDIR="$W/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export XDG_CACHE_HOME="$W/cache"
export CUDA_CACHE_PATH="$W/cache/cuda"
cat "$R/kernelgoal-fullshape-bc800477/stage/"chunk-*.b64 | base64 -d | tar xJm
export CUDA_HOME="$R/kda-astra-high-20261019/venv/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$R/kda-astra-high-20261019/venv/bin:$PATH"
export NVCC="$CUDA_HOME/bin/nvcc"
export TORCH_CUDA_ARCH_LIST=10.0a
export MAX_JOBS=8
export TRITON_CACHE_DIR="$W/triton"
export TORCH_EXTENSIONS_DIR="$W/extensions"
export GDN_BUILD_DIR="$W/gdn-build"
export MOE_BUILD_DIR="$W/moe-build"
export DSA_BUILD_DIR="$W/dsa-build"
export TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas"
export TRITON_PTXAS_BLACKWELL_PATH="$CUDA_HOME/bin/ptxas"
export TRITON_LIBDEVICE_PATH="$CUDA_HOME/nvvm/libdevice/libdevice.10.bc"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
CCCL_TARGET=$(find -L "$CUDA_HOME" /usr/local/cuda/include -path '*/nv/target' -print -quit)
test -n "$CCCL_TARGET"
export CPATH="${CCCL_TARGET%/nv/target}:${CPATH:-}"
export NVCC_PREPEND_FLAGS="-I${CCCL_TARGET%/nv/target}"
mkdir -p "$W/lib"
CUDART=$(find "$CUDA_HOME" -name 'libcudart.so.*' -type f -print -quit)
test -n "$CUDART"
ln -sf "$CUDART" "$W/lib/libcudart.so"
export LIBRARY_PATH="$W/lib:$CUDA_HOME/lib:/usr/local/cuda/lib64/stubs:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$W/lib:$LD_LIBRARY_PATH"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv
nvcc --version
python - <<'VERIFY'
from pathlib import Path
import flashinfer_bench,hashlib
r=Path(flashinfer_bench.__file__).parent
p=r/'bench/config.py'; h=hashlib.sha256(p.read_bytes()).hexdigest(); print('EVAL_HASH',p,h); assert h=='8f2fc3c906b865876d1e69d446eb41a62f32c3dc8ad37411d09b27e95a7212de', 'Evaluator version mismatch'
p=r/'bench/timing.py'; h=hashlib.sha256(p.read_bytes()).hexdigest(); print('EVAL_HASH',p,h); assert h=='c9165db58615afdf03a223120b37418ed5afaf538b4450641f432f53bf68cbe3', 'Evaluator version mismatch'
VERIFY

export W="$PWD"
export PYTHONPATH="$W:/jiuzhang_train/user/chenyaojian/expert-line-audit-c2264b2/packages:${PYTHONPATH:-}"
ln -s /jiuzhang_train/user/chenyaojian/expert-line-audit-c2264b2/dsl46 "$W/dsl46"
ln -s /jiuzhang_train/user/chenyaojian/expert-line-audit-c2264b2/reference_cache "$W/reference_cache"
ln -s /jiuzhang_train/user/chenyaojian/kernelgoal-fa4-tridao-e7a6402/tridao_cache "$W/tridao_cache"
cp /jiuzhang_train/user/chenyaojian/kernelgoal-remainder-03413a0/baseline-topk-r2/*.py "$W/"
cp -r /jiuzhang_train/user/chenyaojian/kernelgoal-remainder-03413a0/baseline-topk-r2/historical-topk "$W/"
python - <<'COPY'
import shutil
shutil.copytree('/jiuzhang_train/user/chenyaojian/kernelgoal-remainder-03413a0/upstream','repository',ignore=shutil.ignore_patterns('trajectory','paper.pdf'))
COPY
selected=0
for assigned_uuid in $(nvidia-smi --query-gpu=uuid --format=csv,noheader); do
 export AUDIT_GPU_UUID=$assigned_uuid
 export CUDA_VISIBLE_DEVICES=$assigned_uuid
 nvidia-smi -i "$assigned_uuid" --reset-gpu-clocks
 if python -u clock_smoke.py selection; then selected=1; break; fi
done
test "$selected" = 1
export PATH="$CUDA_HOME/bin:$PATH"
nvcc --version


printf %s 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 | base64 -d | tar xzm
export DSA_TOPK_BUILD_DIR="$W/dogacel-build"
python - <<'RUN'
import os,subprocess,sys,json,pathlib
exits=[]
for case in ['A','B','C']:
 env=dict(os.environ,TOPK_SHAPE=case)
 log=pathlib.Path('topk-'+case+'.log')
 with log.open('w') as f:
  try:code=subprocess.run([sys.executable,'-u','test_topk_baselines_strict_gpu.py','-v'],env=env,stdout=f,stderr=subprocess.STDOUT,timeout=1200).returncode
  except subprocess.TimeoutExpired:code=124
 exits.append(dict(case=case,returncode=code));print('BASELINE_EXIT',case,code,flush=True)
 if code:print(log.read_text()[-7000:],flush=True)
pathlib.Path('exits.json').write_text(json.dumps(exits))
subprocess.run([sys.executable,'clock_smoke.py','after'],check=True)
RUN
python - <<'ARCHIVE'
from pathlib import Path
import json,zlib,base64
files={str(p):json.loads(p.read_text()) for p in Path('results').glob('*.json')}
files['exits']=json.loads(Path('exits.json').read_text())
blob=base64.b64encode(zlib.compress(json.dumps(files).encode())).decode()
for i in range(0,len(blob),3000):print('ARCHIVE_CHUNK',i//3000,blob[i:i+3000])
print('MARK_ARCHIVE_DONE',(len(blob)+2999)//3000)
ARCHIVE
