"""Probe Triton 3.7.1 feature surface on this image (descriptor API, WS knobs, tutorials)."""
import inspect
import os
import triton
import triton.language as tl

print("triton", triton.__version__)
print("path", os.path.dirname(triton.__file__))

for name in ["make_tensor_descriptor", "async_task", "range", "dot", "atomic_add"]:
    obj = getattr(tl, name, None)
    print(f"--- tl.{name}: {'MISSING' if obj is None else ''}")
    if obj is not None:
        try:
            print("   sig", inspect.signature(obj))
        except Exception as e:
            print("   sig unavailable", e)

print("--- tl namespace (filtered)")
print([n for n in dir(tl) if any(s in n.lower() for s in
      ("desc", "tma", "async", "task", "warp", "spec", "tcgen", "mma", "barrier"))])

print("--- triton.language.extra")
try:
    import triton.language.extra as ex
    print(dir(ex))
except Exception as e:
    print("no extra:", e)

print("--- gluon?")
try:
    import triton.experimental.gluon as gl
    print("gluon present", os.path.dirname(gl.__file__))
    print([n for n in dir(gl) if not n.startswith("_")])
except Exception as e:
    print("no gluon:", e)

print("--- knobs / set_allocator")
print("set_allocator" in dir(triton))
try:
    from triton.tools.tensor_descriptor import TensorDescriptor
    print("TensorDescriptor host-side API present", inspect.signature(TensorDescriptor.from_tensor))
except Exception as e:
    print("no host TensorDescriptor:", e)

print("--- autotune/Config knobs")
print(inspect.signature(triton.Config.__init__))

tut = os.path.join(os.path.dirname(os.path.dirname(triton.__file__)), "triton", "tutorials")
for cand in [tut, "/opt/sglang/lib/python3.12/site-packages/triton/tutorials"]:
    print("tutorials at", cand, os.path.isdir(cand))
    if os.path.isdir(cand):
        print(os.listdir(cand))
