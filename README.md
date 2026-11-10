# Avalon Strix Halo - Linux run fork

> **This is a small run fork of [freriadoc/Avalon](https://github.com/freriadoc/Avalon).**
> All credit for Avalon - the MoE expert caching, the benchmarks, the real documentation -
> goes to [freriadoc](https://github.com/freriadoc). Please read the
> [original repo](https://github.com/freriadoc/Avalon) for what Avalon is and what it
> can do. This fork carries only the port of the 3090 fork's useful bits (heat autosave,
> readable cache stats, vision boot warmup), the Ubuntu ROCm build recipe and a tuned
> launch for one machine: the Minisforum MS-S1 Max, a Strix Halo box. Upstream remains
> the source of truth.

The sibling fork for Windows + RTX 3090 is
**[Kobodera/Avalon3090](https://github.com/Kobodera/Avalon3090)**, and it is where the
vision work and the boot warmup started.

## This project

My daily box is a Minisforum MS-S1 Max: a **Ryzen AI Max+ 395** (32 cores), **128 GB of
unified LPDDR5x** and the **Radeon 8060S** iGPU, 40 CUs of RDNA 3.5. Strix Halo is its
own animal:

- the GPU is **gfx1151**, not the gfx1100 upstream tunes for. Upstream already knows the
  gfx115x family in its arch guards (RDNA 3.5 gets its own branch in MMQ and the GDN
  WMMA kernel compiles for it), so this fork is mostly a build recipe, not new kernels.
- **RAM and VRAM are the same chips.** The BIOS gives the iGPU a 1 GiB visible window
  plus a 61 GiB GTT here; there is no PCIe hop to feed. What the expert cache saves on
  this box is the DRAM-to-DRAM copy per routed expert and the CPU compute of a missed
  expert, not link time. The 49 GiB of experts live in Avalon's own RAM arena and every
  one of them fits, so the disk never matters after boot.
- the CPU is 32 modern cores with AVX2: the experts the GPU does not compute are
  computed much faster than on the 3800XT upstream tunes on.

One warning the memory math earns: the arena the experts register with the driver, the
KV cache, the dense weights and the expert cache all want GPU address space, which on
Strix Halo is the GTT. With 256k context they come to ~65-70 GiB against this BIOS'
61 GiB GTT. Either raise the GTT in the BIOS (the MS-S1 Max lets you), or run 131k
context. The launch script's numbers below are validated against the GTT you have.

## Quick start

Ubuntu 24.04 or 26.04. The ROCm stack is in the Ubuntu repo, no amdgpu install needed:

```bash
sudo apt install -y build-essential git cmake ninja-build python3-pil hipcc \
    libamdhip64-dev libhsa-runtime-dev libhipblas-dev librocblas-dev rocm-device-libs
git clone https://github.com/Kobodera/AvalonStrixHalo.git && cd AvalonStrixHalo
strix/setup.sh
```

`setup.sh` checks the box, builds `llama-server` and `llama-quantize` for gfx1151,
downloads all model files (~96 GB, resumable, SHA256-verified against
`strix/model_manifest.tsv`, same manifest as the 3090 ops repo) and quantizes the vision
projector BF16 -> Q8_0. Re-running after an interrupted download is safe.

Ubuntu's ROCm packaging is the one real build gotcha: cmake's HIP compiler is the raw
llvm clang and it finds neither the ROCm device library nor hipcc's bitcode injection.
`setup.sh` passes `--rocm-device-lib-path` at the resource dir clang ships. By hand:

```bash
llvm_bin=$(hipcc --version | awk '/InstalledDir/ {print $2}')
bitcode=$(ls -d "$llvm_bin"/../lib/clang/*/amdgcn/bitcode | sort -V | tail -1)
ROCM_PATH=/usr cmake -B build_hip -G Ninja -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_HIP_COMPILER="$llvm_bin/clang++" \
    "-DCMAKE_HIP_FLAGS=--rocm-device-lib-path=$bitcode"
cmake --build build_hip -j$(nproc) --target llama-server llama-quantize
```

Then launch the server (256k context + MTP + vision, OpenAI API on port 8080):

```bash
strix/launch_vision_mtp.sh          # AVALON_PORT, AVALON_HOST, AVALON_FAST_BOOT envs
strix/vision_test.sh                # sends a PNG, expects the OCR text back
strix/churn_test.sh                 # IMG/CHAT/TOOLS mixed workload stability gate
```

## What this fork carries on top of upstream

1. **Heat autosave** (`GGML_CUDA_MOE_CACHE_PROFILE_SAVE_INTERVAL_S`, from the 3090
   fork): the expert heat profile persists every minute, so any stop method loses at
   most one interval and the next boot prefills the cache from a fresher ranking.
2. **Readable cache stats** (from the 3090 fork): ggml-cuda INFO lines never reach the
   server log in this build, so the expert cache reports (hits, evictions, fill cost,
   resolved boot config, the host rows state, the timeline diagnostic) now print
   directly to stderr.
3. **Vision boot warmup** (from the 3090 fork): one synthetic 512x512 encode plus a
   decoder pass at startup when `--mmproj` is set. The WDDM first-image deadlock it was
   written for is a Windows thing; on Linux it still warms the vision expert routing,
   which text warmup never touches. Costs ~100 s of boot, saves ~100 s on the first
   image. `GGML_CUDA_MOE_VISION_WARMUP=0` skips it.
4. **The ops scripts** in `strix/` (setup, launch, manifest, tests), the Linux ports of
   Avalon3090ops.

## Measured on the MS-S1 Max

The launch recipe above, greedy decoding, 2 draft MTP tokens, warm expert routing
(Nov 10):

| case | this box | upstream's 7900 XT for scale |
|---|---|---|
| generation, 512 tokens, warm | **38.2 t/s** steady (37.4-38.4 across 10 trials; first request after boot ~27) | 98.4 t/s |
| prefill 12k tokens | 689 t/s | - |
| prefill 48k tokens | 727 t/s | - |
| prefill 190k tokens | 780 t/s | 1,491 t/s |
| boot to serving | 6 min 4 s (first load, cold pages) | - |
| first image after boot, no warmup | 6.5 s, OCR 3/3 | - |

The iGPU has no dedicated memory and less compute than a 7900 XT: on this box the
experts and the dense weights are all read from the same LPDDR5x the CPU writes into,
so the ceiling is memory bandwidth shared with everything, not the PCIe feed upstream
works against. MTP acceptance measured 52 %, mean draft length 2.0.

Stability gates: `vision_test.sh` 3/3 OCR, `churn_test.sh` IMG/CHAT/TOOLS 6/6 pass,
health stays green under churn. GTT peaked at 59.7 of 61.1 GiB with 256k context:
tight, but it fits this BIOS setting. Raise the BIOS GTT before growing context or
the cache budget.

Tuning tried, all measured (each = a full reboot + 3-4 x 512-token trials):
- expert cache: default resolves to **off** here - `--fit` sees the 61 GiB GTT as
  device memory and places every layer on the iGPU, so the scheduler finds no
  host-resident experts to cache. Forcing `GGML_CUDA_MOE_CACHE_MODE=on` with a
  4 GiB budget: no change (37.5-38.3). Off vs default: no change (35.4-37.8).
  On a dGPU the cache avoids PCIe traffic; here the experts sit in the same
  LPDDR5x the GPU reads anyway. Keep the upstream default.
- `-t 32`: much worse (16-22 t/s). Expert rows are read by the GPU through the
  host-mapped buffer; piling CPU threads on the same DRAM only adds contention.
  The default thread count is right.
- `-lm mmap+mlock`: no change (27 / 37.4 / 38.3 / 38.2). The slow first request
  is graph capture, not page eviction, so pinning 36 GiB buys nothing.
- `--spec-draft-n-max 3`: not pursued - chained MTP drafts fall apart, upstream
  validated 2 for this model and MTP head.

## License

MIT, same as [upstream](https://github.com/freriadoc/Avalon). The license stays in the
tree; upstream remains the source of truth for all Avalon code.
