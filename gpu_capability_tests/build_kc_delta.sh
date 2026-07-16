#!/bin/bash

if ! command -v mpicxx &> /dev/null; then
    printf "Error: mpicxx not found. Check your loaded modules.\n"
    exit 1
fi
printf "Using compiler: $(which mpicxx)\n"
printf "MPI version: $(mpirun --version 2>&1 | head -1)\n\n"

# Architecture flags
# Delta CPU nodes: AMD EPYC 7763 "Milan", 64-core, 2.45 GHz
#   https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/architecture.html
# -march=znver3 : Milan-specific tuning (AMD Family 19h)
#   https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html
# -O3           : full optimization
#   https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html
ARCH_FLAGS="-march=znver3 -O3"

init() {
    printf "Setting up directories...\n"
    mkdir -p $HOME/repos $HOME/install/kokkos
    printf "Complete!\n\n"
}

kokkos() {
    if [ -d $HOME/repos/kokkos ]; then
        printf "'kokkos' directory already exists, skipping clone.\n"
    else
        git clone https://github.com/kokkos/kokkos.git $HOME/repos/kokkos
    fi

    rm -rf $HOME/repos/kokkos/build
    mkdir $HOME/repos/kokkos/build

    cmake \
        -S $HOME/repos/kokkos \
        -B $HOME/repos/kokkos/build \
        -DCMAKE_INSTALL_PREFIX=$HOME/install/kokkos \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_COMPILER=mpicxx \
        -DCMAKE_CXX_FLAGS="${ARCH_FLAGS}" \
        -DKokkos_ARCH_ZEN3=ON \
        -DKokkos_ENABLE_OPENMP=ON

    make -C $HOME/repos/kokkos/build -j$(nproc) install
    printf "Kokkos setup complete!\n\n"
}

kokkos-comm() {
    if [ -d $HOME/repos/kokkos-comm ]; then
        printf "'kokkos-comm' directory already exists, updating branch...\n"
        cd $HOME/repos/kokkos-comm
        git fetch origin
        git checkout mpi_windows_old_work
        git pull origin mpi_windows_old_work
    else
        git clone -b mpi_windows_old_work https://github.com/Kaewin/kokkos-comm.git $HOME/repos/kokkos-comm
        cd $HOME/repos/kokkos-comm
    fi

    rm -rf $HOME/repos/kokkos-comm/build
    mkdir $HOME/repos/kokkos-comm/build

    cmake \
        -S $HOME/repos/kokkos-comm \
        -B $HOME/repos/kokkos-comm/build \
        -DCMAKE_CXX_COMPILER=mpicxx \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_FLAGS="${ARCH_FLAGS}" \
        -DKokkos_ROOT=$HOME/install/kokkos \
        -DKokkosComm_ENABLE_TESTS=ON \
        -DKokkosComm_ENABLE_PERFTESTS=ON

    make -C $HOME/repos/kokkos-comm/build -j$(nproc)
    printf "KokkosComm setup complete!\n\n"
}

run_tests() {
    printf "=== Running CTest ===\n"
    cd $HOME/repos/kokkos-comm/build
    ctest -V
}

clean() {
    printf "Cleaning build artifacts...\n"
    rm -rf $HOME/install/kokkos \
           $HOME/repos/kokkos/build \
           $HOME/repos/kokkos-comm/build
    printf "Cleanup complete!\n"
}

if [ "$1" == "clean" ]; then
    clean
else
    set -e
    clean
    init
    kokkos
    kokkos-comm
    run_tests
    printf "Build complete!\n"
fi