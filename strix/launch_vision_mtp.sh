#!/usr/bin/env bash
# Avalon on Strix Halo (gfx1151): MTP draft + vision, on the upstream run line.
# Strix Halo note: RAM and VRAM are the same LPDDR5x (122 GiB visible here).
# --fit places the dense weights on the GPU side and keeps the experts (~49 GiB)
# in Avalon's own RAM arena; the expert cache lives in GPU address space that
# maps the same physical memory. There is no PCIe hop to feed, so what the
# cache saves is the DRAM-to-DRAM copy per routed expert, not link time.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
MODELS="${AVALON_MODEL_DIR:-$(dirname "$REPO")/models/Qwen3.8-Flash-Next-Avalon-GGUF}"
exe="$REPO/build_hip/bin/llama-server"
model="$MODELS/Qwen3.8-Flash-Next-Avalon-IQ3KS-00001-of-00033.gguf"
draft="$MODELS/MTP/mtp-Qwen3.8-Flash-Next-shared-IQ3K-IQ4NL-FR65k.gguf"
mmproj="$MODELS/mmproj-Qwen3.8-Flash-Next-Q8_0.gguf"

for f in "$exe" "$model" "$draft" "$mmproj"; do
    [ -e "$f" ] || { printf 'missing file: %s (run strix/setup.sh first)\n' "$f" >&2; exit 1; }
done

# Heat persistence (this fork's autosave): the file is read at boot and rewritten
# every 60 s, so restarts prefill the cache from the routing actually observed,
# on top of the shipped moe-cache-ranking.bin.
export GGML_CUDA_MOE_CACHE_PROFILE_SAVE="${GGML_CUDA_MOE_CACHE_PROFILE_SAVE:-$MODELS/moe-cache-profile.bin}"

# Warmup at boot: the text pass, plus the ported vision pass when --mmproj is set
# (env names are the ones the 3090 fork uses). The WDDM first-image deadlock the
# warmup was written for is a Windows thing; here it mainly warms the vision
# expert routing. AVALON_FAST_BOOT=1 skips the whole warmup and pays it on the
# first requests instead.
warmup=--warmup
if [ "${AVALON_FAST_BOOT:-0}" = 1 ]; then
    warmup=--no-warmup
    export GGML_CUDA_MOE_VISION_DECODER_WARMUP=0
fi

# -lm mmap: the HIP backend reports mmap_support=false for any iGPU (Strix Halo's
# 8060S included), so load mode "auto" quietly drops mmap and the packed experts
# gather in llama-model-loader then asserts. The gather only needs the file mmap
# on the host side, the copy to the GPU is a normal memcpy: force it.
exec "$exe" -m "$model" \
    -md "$draft" \
    --mmproj "$mmproj" --image-min-tokens 1024 \
    -lm mmap \
    -c 262144 -b 9536 -ub 9536 -fa on -ctk q8_0 -ctv q4_0 --jinja --parallel 1 \
    --spec-type draft-mtp --spec-draft-n-max 2 -ubd 256 -ctkd q8_0 -ctvd q8_0 \
    "$warmup" \
    --host "${AVALON_HOST:-0.0.0.0}" --port "${AVALON_PORT:-8080}" \
    "$@"
