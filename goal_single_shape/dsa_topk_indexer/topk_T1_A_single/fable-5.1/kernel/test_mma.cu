// test_mma.cu — verify mma.sync.aligned fp8 on sm_100
#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdio>

// mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
// Each thread contributes 4 A registers (16 fp8), 2 B registers (8 fp8), 4 C floats
__global__ void test_mma_fp8(float* out) {
    uint32_t ra[4], rb[2];
    float rc[4] = {0.f, 0.f, 0.f, 0.f};

    // fp8 E4M3FN value 1.0 = 0x38; pack 4 per uint32
    uint32_t one4 = 0x38383838u;
    ra[0] = one4; ra[1] = one4; ra[2] = one4; ra[3] = one4;
    rb[0] = one4; rb[1] = one4;

    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};"
        : "=f"(rc[0]), "=f"(rc[1]), "=f"(rc[2]), "=f"(rc[3])
        : "r"(ra[0]), "r"(ra[1]), "r"(ra[2]), "r"(ra[3]),
          "r"(rb[0]), "r"(rb[1]),
          "f"(rc[0]), "f"(rc[1]), "f"(rc[2]), "f"(rc[3])
    );

    if (threadIdx.x == 0) {
        out[0] = rc[0];
        out[1] = rc[1];
        out[2] = rc[2];
        out[3] = rc[3];
    }
}

int main() {
    float* d; cudaMalloc(&d, 4 * sizeof(float));
    test_mma_fp8<<<1, 32>>>(d);
    cudaDeviceSynchronize();
    float h[4]; cudaMemcpy(h, d, 4*sizeof(float), cudaMemcpyDeviceToHost);
    printf("mma result: %f %f %f %f  (expected 32.0)\n", h[0], h[1], h[2], h[3]);
    cudaFree(d);
    return 0;
}
