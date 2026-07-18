#!/usr/bin/env bash
# Usage (host, as of 2026-07-14):
#   KC_SRC=~/repos/kokkos-comm KOKKOS_ROOT=~/installs/kokkos-5.1.99 ./build_smoke.sh
# Usage (CUDA rung B, 2026-07-15):
#   export NVCC_WRAPPER_DEFAULT_COMPILER=CC
#   KC_SRC=~/repos/kokkos-comm KOKKOS_ROOT=~/install/kokkos-cuda \
#     CXX_OVERRIDE=~/repos/kokkos/bin/nvcc_wrapper ./build_smoke.sh
set -euo pipefail
export NVCC_WRAPPER_DEFAULT_COMPILER=CC
: "${KC_SRC:?set KC_SRC to your kokkos-comm checkout}"
: "${KOKKOS_ROOT:?set KOKKOS_ROOT to your Kokkos install prefix}"
# If the header chain wants a generated config header, stub it (harmless if unused):
if grep -rq "cmakedefine" "$KC_SRC/src" 2>/dev/null; then
  mkdir -p /tmp/kc_cfg/KokkosComm
  grep -rl "config.hpp" "$KC_SRC/src" >/dev/null 2>&1 && touch /tmp/kc_cfg/KokkosComm/config.hpp
  EXTRA="-DCMAKE_CXX_FLAGS=-I/tmp/kc_cfg"
fi
# ADDED 2026-07-15 (rung B): optional compiler override for the CUDA toolchain.
if [ -n "${CXX_OVERRIDE:-}" ]; then
  COMPILER_ARG="-DCMAKE_CXX_COMPILER=${CXX_OVERRIDE}"
fi
cmake -S "$(dirname "$0")" -B "$(dirname "$0")/build" \
  -DKC_SRC="$KC_SRC" -DKokkos_ROOT="$KOKKOS_ROOT" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O2 -march=znver3" \
  ${COMPILER_ARG:-} ${EXTRA:-}
cmake --build build -j
echo "Run (inside your allocation):  srun -n 2 ./build/smoke_win_view"
echo "Bridges-2 instead:             srun --mpi=pmix -n 2 ./build/smoke_win_view"
