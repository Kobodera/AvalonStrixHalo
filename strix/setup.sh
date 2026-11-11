#!/usr/bin/env bash
# Setup for AvalonStrixHalo: checks the box, builds llama-server for gfx1151,
# downloads the model set (resumable, SHA256-verified) and quantizes the
# vision projector BF16 -> Q8_0. Re-run any time; finished steps are skipped.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
MODELS="${AVALON_MODEL_DIR:-$(dirname "$REPO")/models/Qwen3.8-Flash-Next-Avalon-GGUF}"
BUILD="$REPO/build_hip"
ARCH="${AMDGPU_TARGETS:-gfx1151}"
FORCE="${FORCE:-0}"

step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
skip() { printf '[skip] %s\n' "$*"; }
die()  { printf '\033[31m[fail] %s\033[0m\n' "$*" >&2; exit 1; }

step "checking the machine"
miss=()
for t in git cmake ninja g++ curl sha256sum python3; do
    command -v "$t" >/dev/null 2>&1 || miss+=("$t")
done
[ -e /usr/lib/x86_64-linux-gnu/cmake/hip/hip-config.cmake ] || miss+=("libamdhip64-dev")
[ "${#miss[@]}" -eq 0 ] || die "missing: ${miss[*]}
install with: sudo apt install -y build-essential git cmake ninja-build hipcc \
    libamdhip64-dev libhsa-runtime-dev libhipblas-dev librocblas-dev rocm-device-libs
(the ROCm stack is packaged by Ubuntu 24.04+ / 26.04, no amdgpu repo needed)"
grep -q avx2 /proc/cpuinfo || die "CPU without AVX2: the CPU expert kernels need it"

gpu_seen=0
for f in /sys/class/drm/card*/device/mem_info_vram_total; do
    [ -e "$f" ] && gpu_seen=1
done
[ "$gpu_seen" = 1 ] || die "no amdgpu DRM device found (is the amdgpu driver loaded?)"

# a seat-logged-in desktop user gets GPU access from logind; a pure SSH session needs the groups
if ! [ -r /dev/kfd ] || ! [ -w /dev/kfd ]; then
    printf '[warn] /dev/kfd is not accessible to this user (a local desktop login gets it from logind; an SSH session does not).\n       fix: sudo usermod -aG render,video "$USER"   then log out and back in\n'
fi

ram_kb=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
[ "$ram_kb" -gt $((56 * 1024 * 1024)) ] || printf '[warn] %d GB RAM: the experts (~49 GiB) want to live fully in RAM\n' $((ram_kb / 1048576))
disk_free=$(df -B1 --output=avail "$(dirname "$REPO")" | tail -1 | tr -d ' ')
[ "$disk_free" -gt $((110 * 1024 * 1024 * 1024)) ] || printf '[warn] less than 110 GB free where the models go\n'
printf '[ok  ] ROCm, AVX2, GPU device, RAM %d GB, %.0f GB free on disk\n' $((ram_kb / 1048576)) "$(echo "$disk_free" | awk '{print $1/1073741824}')"

step "building llama-server + llama-quantize for $ARCH"
if [ -x "$BUILD/bin/llama-server" ] && [ "$FORCE" != 1 ]; then
    skip "build_hip/bin/llama-server exists (FORCE=1 to rebuild)"
else
    rocm=/opt/rocm
    [ -d /opt/rocm ] || rocm=/usr
    hipflags=()
    if [ "$rocm" = /usr ]; then
        # Ubuntu ships the HIP toolchain as llvm-X and the amdgcn bitcode inside its
        # clang resource dir; raw clang finds neither, hipcc injects them. Point cmake
        # at hipcc's clang and pass the bitcode path ourselves.
        llvm_bin=$(hipcc --version 2>/dev/null | awk '/InstalledDir/ {print $2}')
        [ -n "$llvm_bin" ] || die "hipcc not found: install the ROCm apt packages above"
        bitcode=$(ls -d "$llvm_bin"/../lib/clang/*/amdgcn/bitcode 2>/dev/null | sort -V | tail -1)
        [ -n "$bitcode" ] || die "amdgcn bitcode not found near $llvm_bin: install rocm-device-libs"
        hipflags=(-DCMAKE_HIP_COMPILER="$llvm_bin/clang++"
                  "-DCMAKE_HIP_FLAGS=--rocm-device-lib-path=$bitcode")
    fi
    ROCM_PATH="$rocm" cmake -B "$BUILD" -S "$REPO" -G Ninja \
        -DGGML_HIP=ON -DAMDGPU_TARGETS="$ARCH" -DCMAKE_BUILD_TYPE=Release \
        "${hipflags[@]}"
    cmake --build "$BUILD" -j"$(nproc)" --target llama-server llama-quantize
fi

step "downloading the model set (~96 GB, resumable, SHA256-verified)"
AVALON_MODEL_DIR="$MODELS" "$(dirname "$0")/download_model.sh"

step "quantizing the vision projector BF16 -> Q8_0"
mmbf16="$MODELS/staging/mmproj-BF16.gguf"
mmq8="$MODELS/mmproj-Qwen3.8-Flash-Next-Q8_0.gguf"
if [ -f "$mmq8" ]; then
    skip "mmproj Q8_0 already there"
else
    [ -f "$mmbf16" ] || die "staging/mmproj-BF16.gguf missing, re-run the download step"
    "$BUILD/bin/llama-quantize" "$mmbf16" "$mmq8" Q8_0
    sz=$(stat -c%s "$mmq8")
    [ "$sz" -gt 600000000 ] && [ "$sz" -lt 640000000 ] || die "mmproj Q8_0 size unexpected: $sz (want ~616.7 MB)"
fi

step "done"
cat <<EOF
Everything is in place. Start the server with:
    $(dirname "$0")/launch_vision_mtp.sh
It serves an OpenAI-compatible API on http://0.0.0.0:8080.
Check it with: $(dirname "$0")/vision_test.sh
EOF
