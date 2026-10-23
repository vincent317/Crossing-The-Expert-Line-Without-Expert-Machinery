set -e
R=/jiuzhang_train/user/chenyaojian
W="$R/kernelgoal-remainder-03413a0/latest-fa4_bwd-r1"
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
cp /jiuzhang_train/user/chenyaojian/kernelgoal-remainder-03413a0/baseline-fa4-r2/*.py "$W/"
cat /jiuzhang_train/user/chenyaojian/latest-31cb71c-stage/chunk-*.b64 | base64 -d | tar xJm
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

export BASELINE_GROUP=gdn_prefill BASELINE_SHARD=0 BASELINE_SHARDS=4


export PYTHONPATH="/jiuzhang_train/user/chenyaojian/expert-line-audit-c2264b2/fa4deps:/jiuzhang_train/user/chenyaojian/expert-line-audit-c2264b2/dsl452cu13:$PYTHONPATH"
export AUDIT_DSL=/jiuzhang_train/user/chenyaojian/expert-line-audit-c2264b2/dsl452cu13/nvidia_cutlass_dsl/python_packages
export BASELINE_REFERENCE=/jiuzhang_train/user/chenyaojian/expert-line-audit-c2264b2/reference
export LATEST_REVISION=31cb71c7f200c7070a02287578e4a9653f081b3c
python - <<'RUN'
import os,subprocess,sys,json,pathlib
exits=[]
for case in ['kda05_reproduction_single_shape/fable-5.1/fa4_bwd/gqa_b8_n4096_h16_kv2', 'kda05_reproduction_single_shape/fable-5.1/fa4_bwd/mha_b8_n4096_h16', 'kda05_reproduction_single_shape/fable-5.1/fa4_bwd/varlen_t32768_n13_h16']:
 passed=True
 for role in ['baseline','candidate','paired']:
  if role=='paired' and not passed:continue
  env=dict(os.environ,LATEST_CASE=case,LATEST_ROLE=role)
  log=pathlib.Path(case.replace('/','__')+'-'+role+'.log')
  with log.open('w') as f:
   try:code=subprocess.run([sys.executable,'-u','test_latest_single_gpu.py','-v'],env=env,stdout=f,stderr=subprocess.STDOUT,timeout=1800).returncode
   except subprocess.TimeoutExpired:code=124
  passed=passed and code==0
  exits.append(dict(case=case,role=role,returncode=code));pathlib.Path('exits.json').write_text(json.dumps(exits))
  print('CASE_EXIT',case,role,code,flush=True)
  if code:print(log.read_text()[-8000:],flush=True)
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
