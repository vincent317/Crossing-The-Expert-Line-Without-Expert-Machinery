set -e
export PATH=/usr/local/cuda/bin:$PATH
cd /root/work
rm -rf solution && mkdir -p solution
nvcc -arch=sm_100a -cubin -O3 -DNT=512 -DBPSM=2 -DRSEL=3 -DWIN=${WIN:-7} -o solution/moe.cubin src/${SRC:-moe19}.cu
sed -e 's/int(os.environ.get("MOE_NT","512"))/512/' -e 's/min(nb.value, int(os.environ.get("MOE_CTAS_PER_SM","99")))/nb.value/' sol/kernel.py > solution/kernel.py
grep -c "os.environ" solution/kernel.py || true
