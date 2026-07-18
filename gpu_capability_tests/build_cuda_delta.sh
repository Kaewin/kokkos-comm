#!/bin/bash
# build_cuda_delta.sh
# Stage 1 of the device-View work: build a CUDA-enabled Kokkos install on Delta.
# Derived 2026-07-15 from build_kc_delta.sh (patched baseline). COMMIT ME TOO.
#
# What this script deliberately does NOT do:
#   - touch $HOME/install/kokkos or any host build dir (two-binary harness intact)
#   - configure Kokkos Comm at all (headers-only consumption per HANG_DIAGNOSIS.md,
#     2026-07-14: the FetchContent configure hang lives in unit_tests/perf_tests,
#     both gated by KokkosComm_ENABLE_TESTS/PERFTESTS; we never enter that CMake)
#
# Modules required BEFORE running (same set that built rma_dev_test today):
#   module load PrgEnv-gnu cudatoolkit craype-accel-nvidia80
#
# Usage:
#   bash build_cuda_delta.sh         - build + install CUDA Kokkos + smoke test
#   bash build_cuda_delta.sh clean   - remove ONLY the CUDA build/install dirs

KOKKOS_SRC=$HOME/repos/kokkos            # shared source checkout (not modified)
CUDA_BUILD=$HOME/repos/kokkos/build_cuda # NEW build dir; host build/ untouched
CUDA_INSTALL=$HOME/install/kokkos-cuda   # NEW prefix; host install untouched

# Host arch flags carried over from the baseline; device arch is A40 = Ampere
# compute capability 8.6 -> Kokkos_ARCH_AMPERE86. (A100 nodes would be AMPERE80.)
ARCH_FLAGS="-march=znver3 -O3"

# -- Preflight: fail loud and early, one message per missing prerequisite -------
preflight() {
    local ok=1
    command -v nvcc >/dev/null       || { printf "MISSING: nvcc (module load cudatoolkit)\n"; ok=0; }
    command -v CC >/dev/null         || { printf "MISSING: CC wrapper (PrgEnv-gnu)\n"; ok=0; }
    [ -n "$CRAY_ACCEL_TARGET" ]      || printf "WARN: CRAY_ACCEL_TARGET unset (craype-accel-nvidia80 not loaded?) -- GTL wont link into MPI codes\n"
    [ $ok -eq 1 ] || exit 1
    printf "nvcc:      %s\n" "$(which nvcc)"
    printf "CC:        %s\n" "$(which CC)"
    printf "accel:     %s\n\n" "${CRAY_ACCEL_TARGET:-<unset>}"
}

# -- Stage 1: Kokkos, CUDA backend ----------------------------------------------
kokkos_cuda() {
    if [ ! -d $KOKKOS_SRC ]; then
        git clone https://github.com/kokkos/kokkos.git $KOKKOS_SRC
    fi
    # Provenance: record exactly which Kokkos this is. No pull, no checkout --
    # we build whatever the working tree holds and we write down what that was.
    printf "Kokkos source state:\n"
    git -C $KOKKOS_SRC rev-parse HEAD
    git -C $KOKKOS_SRC describe --tags --always
    printf "\n"

    rm -rf $CUDA_BUILD
    mkdir -p $CUDA_BUILD

    # nvcc_wrapper (ships inside the Kokkos repo) is the CXX compiler; it hands
    # host-side compilation to CC (Cray wrapper -> gcc + MPI + GTL) and device
    # code to nvcc. This is the standard Cray-system Kokkos recipe.
    export NVCC_WRAPPER_DEFAULT_COMPILER=CC

    cmake \
        -S $KOKKOS_SRC \
        -B $CUDA_BUILD \
        -DCMAKE_INSTALL_PREFIX=$CUDA_INSTALL \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_COMPILER=$KOKKOS_SRC/bin/nvcc_wrapper \
        -DCMAKE_CXX_FLAGS="${ARCH_FLAGS}" \
        -DKokkos_ARCH_ZEN3=ON \
        -DKokkos_ARCH_AMPERE86=ON \
        -DKokkos_ENABLE_CUDA=ON \
        -DKokkos_ENABLE_CUDA_LAMBDA=ON \
        -DKokkos_ENABLE_OPENMP=ON

    make -C $CUDA_BUILD -j$(nproc) install
    printf "CUDA Kokkos install complete: %s\n\n" "$CUDA_INSTALL"
}

# -- Smoke: prove the install can compile and run a device parallel_reduce ------
smoke() {
    local d=$CUDA_BUILD/smoke
    mkdir -p $d
    cat > $d/smoke.cpp << 'EOF'
#include <Kokkos_Core.hpp>
#include <cstdio>
int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    const long N = 1 << 20;
    Kokkos::View<double*, Kokkos::CudaSpace> v("v", N);
    Kokkos::parallel_for("fill", N, KOKKOS_LAMBDA(const long i) { v(i) = double(i); });
    double sum = 0.0;
    Kokkos::parallel_reduce("sum", N,
      KOKKOS_LAMBDA(const long i, double& acc) { acc += v(i); }, sum);
    const double expected = double(N) * (N - 1) / 2.0;
    std::printf("smoke: sum=%.10e expected=%.10e -> %s\n",
                sum, expected, sum == expected ? "PASS" : "FAIL");
  }
  Kokkos::finalize();
  return 0;
}
EOF
    cat > $d/CMakeLists.txt << 'EOF'
cmake_minimum_required(VERSION 3.23)
project(smoke CXX)
find_package(Kokkos REQUIRED)
add_executable(smoke smoke.cpp)
target_link_libraries(smoke PRIVATE Kokkos::kokkos)
EOF
    cmake -S $d -B $d/build -DKokkos_ROOT=$CUDA_INSTALL \
          -DCMAKE_CXX_COMPILER=$KOKKOS_SRC/bin/nvcc_wrapper \
          -DCMAKE_BUILD_TYPE=Release
    cmake --build $d/build -j
    printf "Smoke binary: %s/build/smoke\n" "$d"
    printf "Run it on a GPU node:  srun -N1 -n1 --gpus-per-node=1 %s/build/smoke\n\n" "$d"
}

clean() {
    printf "Removing CUDA build/install dirs ONLY (host harness untouched)...\n"
    rm -rf $CUDA_BUILD $CUDA_INSTALL
    printf "Done.\n"
}

if [ "$1" == "clean" ]; then
    clean
else
    set -e
    preflight
    kokkos_cuda
    smoke
    printf "Stage 1 complete. Stage 2 = wrapper-arm test built the arm4_smoke way:\n"
    printf "  headers from \$HOME/repos/kokkos-comm/src, Kokkos_ROOT=%s\n" "$CUDA_INSTALL"
fi
