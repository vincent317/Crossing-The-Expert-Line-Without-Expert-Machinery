"""JIT build of the custom tcgen05 FMHA extension.

-arch=sm_100a is NOT enough on CUDA 13: nvcc silently degrades it to sm_100,
where tcgen05 is unavailable. The -gencode form is required.
"""

import os

from torch.utils.cpp_extension import load


def build(verbose=True):
    here = os.path.dirname(os.path.abspath(__file__))
    vlayout = os.environ.get("VLAYOUT", "1")
    block_n = os.environ.get("BLOCK_N", "128")
    rescale_thresh = os.environ.get("RESCALE_THRESH", "0.0f")
    cpasync = os.environ.get("USE_CPASYNC", "0")
    ablate = os.environ.get("ABLATE", "0")
    nkv = os.environ.get("NSTAGE_KV", "5")
    ns = os.environ.get("NSTAGE_S", "3")
    tag = (f"v{vlayout}_n{block_n}_r{rescale_thresh.rstrip('f').replace('.', 'p')}"
           f"_c{cpasync}_a{ablate}_kv{nkv}_s{ns}")
    build_dir = os.environ.get("TORCH_EXT_DIR", "/tmp/ext") + f"_{tag}"
    os.makedirs(build_dir, exist_ok=True)
    return load(
        name=f"fmha_tcgen05_{tag}",
        sources=[os.path.join(here, "binding.cu"), os.path.join(here, "fmha.cu")],
        extra_cuda_cflags=[
            "-O3",
            "-gencode=arch=compute_100a,code=sm_100a",
            "--use_fast_math",
            "-lineinfo",
            f"-DVLAYOUT={vlayout}",
            f"-DBLOCK_N={block_n}",
            f"-DRESCALE_THRESH={rescale_thresh}",
            f"-DUSE_CPASYNC={cpasync}",
            f"-DABLATE={ablate}",
            f"-DNSTAGE_KV={nkv}",
            f"-DNSTAGE_S={ns}",
            "--ptxas-options=-v",
        ],
        extra_cflags=["-O3"],
        verbose=verbose,
        build_directory=build_dir,
    )
