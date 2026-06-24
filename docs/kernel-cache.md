# opt-in baked compiled-kernel cache

the worker image (`Dockerfile.worker`) can ship a pre-compiled kernel cache so a cold worker skips
the ~10-15 min first-use JIT that #194 reintroduced. it is fully opt-in: the default build bakes
nothing and behaves exactly as before.

measured win: cold compile ~124s -> warm load ~0.2s (537x).

## how it works

- the image always sets `TRITON_CACHE_DIR=/opt/flash/kernelcache/triton` and
  `TORCHINDUCTOR_CACHE_DIR=/opt/flash/kernelcache/inductor`, and creates those dirs.
- when built with `--build-arg BUILD_KERNEL_CACHE=true`, a portable mega-cache produced on a real
  GPU is baked into `/opt/flash/kernelcache` via a conditional `COPY build/kernel_cache/`. that dir
  is checked into the repo with just a `.keep` placeholder so the `COPY` source always exists (an
  empty glob is a hard build error); the guard only bakes when `mega_cache.bin` is actually present
  and the build-arg is set, so a missing artifact is a no-op.
- at boot, `flash.engine.worker._load_kernel_cache_if_present()` checks for the baked blob at
  `/opt/flash/kernelcache/mega_cache.bin` and, if present, calls
  `torch.compiler.load_cache_artifacts()` to restore the compiled kernels before any model import.
  this is best-effort and never raises.

the hot kernels covered: flash-attn fwd/bwd, the liger fused cross-entropy, flash-linear-attention's
gated-deltanet (qwen3.5/3.6 hybrid), and a representative `torch.compile` (torchinductor).

## producing the cache (on a GPU builder)

kernels are content-addressed by GPU arch + toolchain version, so the cache MUST be compiled on a
real GPU of the target arch (the CI image builder normally has none). on a GPU runner:

```sh
# 1. warm the kernels and write the portable mega-cache
python -m flash.engine.worker.kernel_warmup --arch 9.0 --out build/kernel_cache

# 2. build the image with the bake enabled (build/kernel_cache/ is copied in)
docker build -f Dockerfile.worker --build-arg BUILD_KERNEL_CACHE=true \
  -t ghcr.io/freesolo-co/flash-worker:cu128 .
```

`kernel_warmup.py` is import-safe without torch (it `py_compile`s on the CPU-only CI image); every
heavy import lives inside a function and each warm step is independently guarded, so a kernel that
can't compile is skipped and the bake saves whatever did compile. `--arch` pins
`TORCH_CUDA_ARCH_LIST`; `--out` sets the cache dir (default `/opt/flash/kernelcache`).

## fallback (the default)

with `BUILD_KERNEL_CACHE=false` (the default) nothing is baked and the build needs no GPU. the
worker JITs the kernels on first use exactly as before; flash #163's init heartbeat keeps the
control-plane stall detector quiet through that compile. an absent or unreadable cache is a no-op at
boot, so the opt-in is purely additive.
