#!/bin/bash

init() {
    printf "Setting up directories...\n"
    mkdir -p $HOME/repos $HOME/install/kokkos
    printf "Complete!\n"
}

kokkos() {
    if [ -d $HOME/repos/kokkos ]; then
        printf "'kokkos' directory already exists.\n"
    else
        git clone https://github.com/kokkos/kokkos.git $HOME/repos/kokkos
    fi
    rm -rf $HOME/repos/kokkos/build; mkdir $HOME/repos/kokkos/build
    cmake \
        -S $HOME/repos/kokkos \
        -B $HOME/repos/kokkos/build \
        -DCMAKE_INSTALL_PREFIX=$HOME/install/kokkos \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_COMPILER=g++;
    make -C $HOME/repos/kokkos/build -j$(nproc) install
    printf "Kokkos setup complete!\n\n"
}

kokkos-comm(){
    if [ -d $HOME/repos/kokkos-comm ]; then
        printf "'kokkos-comm' directory already exists.\n"
        cd $HOME/repos/kokkos-comm
        git fetch origin  # Make sure we have the latest remote info
        git checkout mpi_windows
        git pull origin mpi_windows  # Update to latest changes on the branch
    else
        # Clone and checkout the specific branch
        git clone -b mpi_windows https://github.com/Kaewin/kokkos-comm.git $HOME/repos/kokkos-comm
        cd $HOME/repos/kokkos-comm
    fi
    
    rm -rf $HOME/repos/kokkos-comm/build; mkdir $HOME/repos/kokkos-comm/build
    cmake \
        -S $HOME/repos/kokkos-comm \
        -B $HOME/repos/kokkos-comm/build \
        -DCMAKE_CXX_COMPILER=g++ \
        -DKokkos_ROOT=$HOME/install/kokkos \
        -DKokkosComm_ENABLE_TESTS=ON;
    make -C $HOME/repos/kokkos-comm/build -j$(nproc)
    printf "KokkosComm setup complete!\n\n"
}

clean() {
    printf "Cleaning up...\n"
    rm -rf $HOME/install/kokkos $HOME/repos/kokkos/build $HOME/repos/kokkos-comm/build
    printf "Cleanup complete!\n"
}

test(){
    cd $HOME/repos/kokkos-comm/build; ctest -V
}

# Check for arguments
if [ "$1" == "clean" ]; then
    clean
else
    set -e
    clean; init; kokkos; kokkos-comm; test
fi
