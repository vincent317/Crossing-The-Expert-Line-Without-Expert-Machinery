CI=/opt/sglang/lib/python3.12/site-packages/flashinfer/data/cutlass/include
echo "### where is tcgen05.mma emitted"
grep -rln "tcgen05.mma" $CI/cute/ | head
echo "### raw tcgen05.mma instruction forms (unique)"
grep -rho "tcgen05\.mma[a-zA-Z0-9_.:]*" $CI/cute/ | sort -u | head -30
echo "### one full asm block for kind::f16 SS"
grep -rn -A 30 'tcgen05.mma.cta_group::1.kind::f16' $CI/cute/arch/mma_sm100_umma.hpp 2>/dev/null | head -50
echo "### fallback: search all files for the asm literal with context"
for f in $(grep -rln "tcgen05.mma" $CI/cute/); do echo "--- $f"; grep -n -B 6 -A 12 'tcgen05.mma' $f | head -60; break; done
