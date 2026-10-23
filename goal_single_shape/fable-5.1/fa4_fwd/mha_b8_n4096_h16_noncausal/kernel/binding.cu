// Torch extension entry point for the custom tcgen05 FMHA forward kernel.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>

extern "C" __global__ void fmha_fwd(const __nv_bfloat16 *, const __nv_bfloat16 *,
                                    const __nv_bfloat16 *, __nv_bfloat16 *, int,
                                    int, float);
extern "C" size_t fmha_smem_bytes();

#define BLOCK_M_H 128

at::Tensor fmha_forward(at::Tensor q, at::Tensor k, at::Tensor v,
                        double softmax_scale) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kBFloat16, "q must be cuda bf16");
  TORCH_CHECK(q.dim() == 4, "q must be (B,S,H,D)");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
              "q/k/v must be contiguous");
  const int B = q.size(0), S = q.size(1), H = q.size(2), D = q.size(3);
  TORCH_CHECK(D == 128, "head_dim must be 128");
  TORCH_CHECK(S % 128 == 0, "seqlen must be a multiple of 128");

  auto o = at::empty_like(q);

  // fold log2(e) into the scale so the kernel can use exp2
  const float scale_log2e = (float)(softmax_scale * 1.4426950408889634);

  static bool smem_configured = false;
  const int smem = (int)fmha_smem_bytes();
  if (!smem_configured) {
    cudaFuncSetAttribute((void *)fmha_fwd,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    int nb = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, fmha_fwd, 128, smem);
    printf("[fmha] dynamic smem=%d bytes, max active CTAs/SM=%d\n", smem, nb);
    smem_configured = true;
  }

  dim3 grid(S / BLOCK_M_H, B * H);
  fmha_fwd<<<grid, 128, smem, at::cuda::getCurrentCUDAStream()>>>(
      (const __nv_bfloat16 *)q.data_ptr(), (const __nv_bfloat16 *)k.data_ptr(),
      (const __nv_bfloat16 *)v.data_ptr(), (__nv_bfloat16 *)o.data_ptr(), S, H,
      scale_log2e);
  C10_CUDA_CHECK(cudaGetLastError());
  return o;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fwd", &fmha_forward, "custom tcgen05 flash attention forward");
}
