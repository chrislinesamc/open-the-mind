#!/usr/bin/env bash
# One-command trial on public BigBrain data: prints this machine's specs, checks that it can run the viewer,
# installs the libraries that fit it (PyTorch for its GPU driver) into .venv, downloads the data, converts it
# to a Zarr pyramid within a memory budget and starts the viewer. Steps whose result already exists are
# skipped, so the script can be re-run after an interruption.
#
# usage: test/test.sh [nifti|png|slab]
#   nifti  the whole brain as one 100 um NIfTI volume (1392 x 1541 x 1209): one 0.6 GB download   [default]
#          (scripts/nifti_to_zarr.py)
#   png    the whole brain from its 7,404 coronal PNG thumbnails (500 x 434 pixels, 0.43 GB): 20 um between
#          sections, 278 um pixels (scripts/png_to_zarr.py)
#   slab   64 full-resolution 20 um sections from the middle of the brain (6572 x 5711, 0.9 GB)
# settings (environment variables, defaults in brackets):
#   PORT [1729]  VENV [.venv]
#   BIND [0.0.0.0]              address the viewer listens on: 0.0.0.0 = all interfaces (needed inside Docker,
#                               with the port published, e.g. docker run --gpus all -p 1729:1729 ...); 127.0.0.1 = this machine only
#   MEM_GB [2]                  RAM and GPU memory budget (each) of the conversion
#   POOL_GB [1]  HOST_GB [0.5]  GPU brick pool and host brick cache of the viewer (under 2 GB VRAM and 2 GB RAM)
#   SECTIONS [64]  FIRST [3700] sections of the slab test
#   TORCH_INDEX                 force a PyTorch wheel index, e.g. cu126, cu128, cu130
set -euo pipefail
cd "$(dirname "$0")/.."

MODE=${1:-nifti}
SECTIONS=${SECTIONS:-64}
FIRST=${FIRST:-3700}
PORT=${PORT:-1729}
BIND=${BIND:-0.0.0.0}
VENV=${VENV:-.venv}
POOL_GB=${POOL_GB:-1}
HOST_GB=${HOST_GB:-0.5}
MEM_GB=${MEM_GB:-2}
MIN_CC=7.5                                   # oldest GPU generation PyTorch's current CUDA wheels run on (Turing)
FTP=https://ftp.bigbrainproject.org/bigbrain-ftp/BigBrainRelease.2015
case "$MODE" in                              # data folder and the disk space a fresh run needs (GB)
    nifti) DATA=data/bigbrain_100um;  DATA_GB=4 ;;    # 0.6 download + 2.6 raw while converting + 0.7 pyramid
    png)   DATA=data/bigbrain_thumbs; DATA_GB=1 ;;
    slab)  DATA=data/bigbrain_slab;   DATA_GB=2 ;;
    *) echo "usage: test/test.sh [nifti|png|slab]" >&2; exit 2 ;;
esac
ZARR=$DATA.ome.zarr
PLANES=("sagittal plane (L–R, x)" "axial plane (S–I, y)" "coronal plane (A–P, z, section axis)")

step() { printf '\n== %s\n' "$*"; }
die() { printf '\nerror: %s\n' "$*" >&2; exit 1; }
ge() { awk -v a="$1" -v b="$2" 'BEGIN { exit !(a + 0 >= b + 0) }'; }      # numeric a >= b

step "1/6 device"
OS=$(uname -s); ARCH=$(uname -m)
DISTRO=$( (. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME") || echo "$OS")
CPU=$(lscpu 2>/dev/null | sed -n 's/^Model name: *//p' | head -1)
CORES=$(nproc 2>/dev/null || echo "?")
RAM_TOTAL=$(awk '/MemTotal/ { printf "%.1f", $2 / 2^20 }' /proc/meminfo 2>/dev/null || echo 0)
RAM_FREE=$(awk '/MemAvailable/ { printf "%.1f", $2 / 2^20 }' /proc/meminfo 2>/dev/null || echo 0)
DISK_FREE=$(df -Pk . | awk 'NR == 2 { printf "%.1f", $4 / 2^20 }')
PY=${PYTHON:-$(command -v python3.12 || command -v python3 || true)}
PYV=$([ -n "$PY" ] && "$PY" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")' || echo "none")
GPU="" CC="" VRAM_TOTAL=0 VRAM_FREE=0 DRIVER="" CUDA=""
if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
    IFS=, read -r GPU VRAM_TOTAL VRAM_FREE DRIVER < <(nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version \
        --format=csv,noheader,nounits 2>/dev/null | head -1 | sed 's/, */,/g') || true
    CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | grep -E '^[0-9]+\.[0-9]+$' || true)   # drivers before 2022 lack it
    CUDA=$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9]*\.[0-9]*\).*/\1/p' | head -1)
    VRAM_TOTAL=$(awk -v m="$VRAM_TOTAL" 'BEGIN { printf "%.1f", m / 1024 }'); VRAM_FREE=$(awk -v m="$VRAM_FREE" 'BEGIN { printf "%.1f", m / 1024 }')
fi
printf '  %-8s %s\n' "system" "$DISTRO ($OS $(uname -r), $ARCH)" "CPU" "${CPU:-unknown}, $CORES threads" \
    "RAM" "$RAM_TOTAL GB, $RAM_FREE GB available" "disk" "$DISK_FREE GB free here" "Python" "${PY:-not found} ($PYV)"
if [ -n "$GPU" ]; then
    printf '  %-8s %s\n' "GPU" "$GPU, compute capability ${CC:-unknown}" "VRAM" "$VRAM_TOTAL GB, $VRAM_FREE GB free" \
        "driver" "$DRIVER, supports CUDA ${CUDA:-unknown}"
else
    printf '  %-8s %s\n' "GPU" "no NVIDIA GPU found"
fi

step "2/6 compatibility"
BAD=0
check() {   # check <1 if ok> <label> <what to do if not>
    if [ "$1" = 1 ]; then printf '  ✓ %s\n' "$2"; else printf '  ✗ %s: %s\n' "$2" "$3"; BAD=1; fi
}
need_disk=$DATA_GB; [ -x "$VENV/bin/python" ] || need_disk=$((need_disk + 6)); [ -f "$ZARR/zarr.json" ] && need_disk=$((need_disk - DATA_GB))
check "$([ "$OS" = Linux ] && echo 1)" "Linux" "the viewer's GPU kernels (Triton) run on Linux only (WSL2 on Windows works)"
check "$([[ "$ARCH" =~ ^(x86_64|aarch64)$ ]] && echo 1)" "64-bit x86 or ARM CPU" "$ARCH has no PyTorch CUDA builds"
check "$([ -n "$GPU" ] && echo 1)" "NVIDIA GPU" "the viewer renders on an NVIDIA GPU with CUDA; this machine has none, so it cannot run the viewer"
if [ -n "$GPU" ]; then
    check "$( { [ -z "$CC" ] || ge "$CC" "$MIN_CC"; } && echo 1)" "GPU generation (compute capability $MIN_CC or newer)" \
        "$GPU is $CC: too old for current PyTorch CUDA builds (GTX 16xx / RTX 20xx or newer is needed)"
    check "$([ -n "$CUDA" ] && ge "$CUDA" 12.6 && echo 1)" "driver supports CUDA 12.6 or newer" \
        "the driver supports CUDA ${CUDA:-unknown}: update the NVIDIA driver"
    check "$(ge "$VRAM_FREE" 2 && echo 1)" "2 GB GPU memory free" "only $VRAM_FREE GB free: close other GPU programs"
fi
check "$(ge "$RAM_FREE" 2 && echo 1)" "2 GB RAM available" "only $RAM_FREE GB available: close other programs"
if [ "$need_disk" -gt 0 ]; then
    check "$(ge "$DISK_FREE" "$need_disk" && echo 1)" "$need_disk GB disk free (libraries and data)" "only $DISK_FREE GB free"
else
    check 1 "disk space (libraries and data already here)"
fi
check "$([ "$PYV" != none ] && ge "$PYV" 3.10 && "$PY" -c 'import venv, ensurepip' 2>/dev/null && echo 1)" "Python 3.10 or newer with venv" \
    "install python3 and python3-venv (e.g. sudo apt install python3-venv)"
check "$( (command -v cc || command -v gcc) >/dev/null && echo 1)" "C compiler (Triton builds a small launcher)" "sudo apt install build-essential"
HEADERS=0
if [ "$PYV" != none ] && "$PY" -c 'import sysconfig, os, sys; sys.exit(not os.path.exists(os.path.join(sysconfig.get_paths()["include"], "Python.h")))'; then
    HEADERS=1
elif [ "$PYV" = 3.12 ] && { [ -f data/py312-headers/usr/include/python3.12/Python.h ] || command -v apt-get >/dev/null; }; then
    HEADERS=2                                # missing, but can be unpacked without root in step 3
fi
check "$([ "$HEADERS" != 0 ] && echo 1)" "Python headers$([ "$HEADERS" = 2 ] && echo " (a copy is used, see step 3)")" \
    "sudo apt install python$PYV-dev"
[ "$BAD" = 0 ] || die "this machine cannot run the viewer: fix the items marked ✗ above and run again"

if [ -z "${TORCH_INDEX:-}" ]; then           # newest PyTorch wheel index the driver can run
    TORCH_INDEX=$(awk -v v="$CUDA" 'BEGIN { split(v, a, "."); c = a[1] * 100 + a[2];
        if (c >= 1300) print "cu130"; else if (c >= 1208) print "cu128"; else print "cu126"; }')
fi
echo "  compatible: PyTorch wheels for $TORCH_INDEX ($ARCH)"

step "3/6 installing the libraries into $VENV"
if [ "$HEADERS" = 2 ] && [ ! -f data/py312-headers/usr/include/python3.12/Python.h ]; then
    # Triton compiles against Python.h: unpack the -dev packages into data/py312-headers, where
    # server/raycast.py looks for them (no root needed)
    echo "unpacking python3.12-dev into data/py312-headers"
    tmp=$(mktemp -d)
    (cd "$tmp" && apt-get download -q libpython3.12-dev python3.12-dev) || die "apt-get download failed: sudo apt install python3.12-dev"
    mkdir -p data && for d in "$tmp"/*.deb; do dpkg-deb -x "$d" data/py312-headers; done
    rm -rf "$tmp"
fi
[ -x "$VENV/bin/python" ] || "$PY" -m venv "$VENV" || die "could not create $VENV (sudo apt install python3-venv)"
VPY="$VENV/bin/python"
if ! "$VPY" -c 'import torch, torchvision, triton; assert torch.cuda.is_available()' 2>/dev/null; then
    "$VPY" -m pip install -q --upgrade pip
    "$VPY" -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/$TORCH_INDEX"
fi
"$VPY" -m pip install -q -r requirements.txt
"$VPY" -c 'import torch, triton; assert torch.cuda.is_available(), "PyTorch cannot see the GPU"
print(f"torch {torch.__version__} (CUDA {torch.version.cuda}), triton {triton.__version__}, {torch.cuda.get_device_name(0)}")'

step "4/6 downloading BigBrain ($MODE)"
mkdir -p data
if [ -f "$ZARR/zarr.json" ]; then
    echo "not needed: $ZARR is already built"
else
    case "$MODE" in
        nifti) if [ -f "$DATA.nii.gz" ]; then
                   echo "$DATA.nii.gz already downloaded"
               else                          # resumable: re-running continues the .part file
                   curl -fL -C - --retry 5 -o "$DATA.nii.gz.part" "$FTP/3D_Volumes/Histological_Space/nii/full8_100um_optbal.nii.gz"
                   mv "$DATA.nii.gz.part" "$DATA.nii.gz"
               fi ;;
        png)   "$VPY" scripts/fetch_ftp_sections.py "$DATA" --url "$FTP/2D_Final_Sections/Coronal/Png/Thumbnail/" --workers 16 ;;
        slab)  "$VPY" test/fetch_sample_sections.py "$DATA" --first "$FIRST" --count "$SECTIONS" ;;
    esac
fi

step "5/6 converting to Zarr (memory budget $MEM_GB GB)"
if [ -f "$ZARR/zarr.json" ]; then            # written last: its presence means the conversion finished
    echo "$ZARR already built"
else
    rm -rf "$ZARR"
    case "$MODE" in
        nifti) "$VPY" scripts/nifti_to_zarr.py "$DATA.nii.gz" "$ZARR" --mem-gb "$MEM_GB" ;;
        png)   "$VPY" scripts/png_to_zarr.py "$DATA" "$ZARR" --section-um 20 --pixel-um 278.2 --pixel-um-y 278.5 \
                   --planes "${PLANES[@]}" --mem-gb "$MEM_GB" ;;
        slab)  "$VPY" scripts/png_to_zarr.py "$DATA" "$ZARR" --section-um 20 --pixel-um 21.1667 \
                   --first "$FIRST" --count "$SECTIONS" --planes "${PLANES[@]}" --mem-gb "$MEM_GB" ;;
    esac
fi

step "6/6 starting the viewer"
IP=$(hostname -I 2>/dev/null | awk '{print $1}')
if [ "$BIND" = 0.0.0.0 ]; then
    echo "open  http://localhost:$PORT/${IP:+   (from another machine: http://$IP:$PORT/)}"
    [ -f /.dockerenv ] && echo "      inside Docker: publish the port (docker run -p $PORT:$PORT ...) and open http://<docker host>:$PORT/"
else
    echo "open  http://$BIND:$PORT/"
fi
echo "stop  Ctrl+C"
exec "$VPY" server/app.py "$ZARR" "$PORT" "$BIND" --pool-gb "$POOL_GB" --host-gb "$HOST_GB" \
    --staging-mb 64 --threads 8 --pin-share 0.25
