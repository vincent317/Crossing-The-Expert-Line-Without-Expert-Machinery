export PIP_CONSTRAINT=
pipx() { for i in 1 2 3 4 5; do pip install --no-input -q "$@" && return 0; sleep 8; done; return 1; }
pipx apache-tvm-ffi >/dev/null 2>&1; pipx "nvidia-cutlass-dsl[cu13]==4.7.1" >/dev/null 2>&1
pipx -U nvidia-cudnn-frontend >/dev/null 2>&1; pipx --no-deps flashinfer-python >/dev/null 2>&1
for i in $(seq 1 12); do ERR=$(python -c "import flashinfer" 2>&1); [ -z "$ERR" ] && break
  MOD=$(echo "$ERR" | grep -oE "No module named '[^']+'" | head -1 | sed "s/No module named '//;s/'//" | cut -d. -f1)
  [ -n "$MOD" ] && pipx "$MOD" >/dev/null 2>&1 || break; done
pipx "triton==3.5.0" >/dev/null 2>&1
python -u -c "import triton,torch;print('triton',triton.__version__,'torch',torch.__version__,torch.version.cuda)"
python -u final_run.py "$CFGS" 2>&1
