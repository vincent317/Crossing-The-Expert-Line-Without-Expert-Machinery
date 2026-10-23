CI=/opt/sglang/lib/python3.12/site-packages/flashinfer/data/cutlass/include
echo "### SmemDescriptor"
grep -n -A 40 "union SmemDescriptor" $CI/cute/arch/mma_sm100_desc.hpp | head -60
echo "### make_smem_desc / LayoutType"
grep -n -B3 -A 30 "enum class LayoutType" $CI/cute/arch/mma_sm100_desc.hpp | head -50
echo "### tcgen05.mma asm strings (kind::f16, cta_group 1 and 2)"
grep -n -B12 "tcgen05.mma.cta_group::1.kind::f16" $CI/cute/arch/mma_sm100.hpp | head -80
echo "### tcgen05 alloc/dealloc/ld/st/fence asm"
grep -rn "tcgen05.alloc\|tcgen05.dealloc\|tcgen05.relinquish\|tcgen05.fence\|tcgen05.wait\|tcgen05.commit" $CI/cute/arch/tmem_allocator_sm100.hpp $CI/cute/arch/*sm100*.hpp 2>/dev/null | head -30
echo "### tcgen05.ld/st variants present"
grep -rho "tcgen05\.\(ld\|st\)[a-z0-9_.:]*" $CI/cute/arch/ | sort -u | head -40
