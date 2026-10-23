echo "### nvcc supported gpu-archs"
nvcc --list-gpu-arch 2>&1 | tr '\n' ' '; echo
nvcc --list-gpu-code 2>&1 | tr '\n' ' '; echo
echo "### try variants"
for A in "-arch=sm_100a" "-gencode arch=compute_100a,code=sm_100a" "-arch=sm_110a" "-gencode arch=compute_103a,code=sm_103a" "-arch=native"; do
  printf "%-45s : " "$A"
  nvcc $A -O3 -c smoke_umma.cu -o /tmp/t.o 2>&1 | grep -c "error" | tr '\n' ' '
  nvcc $A -O3 -c smoke_umma.cu -o /tmp/t.o 2>&1 | grep -m1 "error" || echo OK
done
echo "### device arch"
python -c "import torch;print(torch.cuda.get_device_capability(0), torch.cuda.get_device_properties(0))"
