set -e
CI=/opt/sglang/lib/python3.12/site-packages/flashinfer/data/cutlass/include
grep -m1 -r "CUTLASS_MAJOR\|#define CUTLASS_VERSION" $CI/cutlass/version.h || true
echo "=== version.h ==="; sed -n '1,60p' $CI/cutlass/version.h | grep -i "define" || true
echo "=== FILES ==="; ls $CI/cute/arch/ | grep -i "sm100\|tmem" 
echo "=== mma_sm100_desc.hpp ==="; cat $CI/cute/arch/mma_sm100_desc.hpp
