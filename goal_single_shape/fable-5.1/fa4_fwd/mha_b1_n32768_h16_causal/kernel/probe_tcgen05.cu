// Smoke test: do tcgen05 PTX instructions assemble for sm_100a?
#include <cstdio>
#include <cstdint>
__global__ void k(unsigned *smem_out) {
  __shared__ __align__(1024) unsigned smem[64];
  if (threadIdx.x == 0) smem[0] = 0;
  __syncthreads();
  unsigned tmem_addr;
  asm volatile("{ .reg .b32 a; tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 128; }" ::"l"((unsigned long long)(uintptr_t)&smem[0]));
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::);
  tmem_addr = smem[0];
  unsigned long long desc_a = 0, desc_b = 0;
  unsigned idesc = 0;
  asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, 0;" ::"r"(tmem_addr), "l"(desc_a), "l"(desc_b), "r"(idesc));
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"l"((unsigned long long)(uintptr_t)&smem[2]));
  unsigned r[4];
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(tmem_addr));
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 128;" ::"r"(tmem_addr));
  smem_out[0] = r[0];
}
int main(){ printf("compiled\n"); return 0; }
