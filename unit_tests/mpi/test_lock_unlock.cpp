//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2025) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#include <gtest/gtest.h>
#include <KokkosComm/KokkosComm.hpp>

namespace {

using namespace KokkosComm;

// Test: Basic Lock/Unlock Exclusive Put
TEST(LockUnlockTest, ExclusivePut) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        GTEST_SKIP() << "This test requires at least 2 MPI processes";
    }

    Kokkos::View<double*, Kokkos::HostSpace> data("data", 1);
    data(0) = static_cast<double>(rank);
    
    Window<decltype(data)> window(data, MPI_COMM_WORLD);
    
    if (rank == 0) {
        window.lock(Window<decltype(data)>::LockType::Exclusive, 1);
        
        double value = 42.0;
        window.put(&value, 1, 1, 0);
        
        window.unlock(1);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 1) {
        EXPECT_EQ(data(0), 42.0);
    }
}

// Test: Shared Lock Get
TEST(LockUnlockTest, SharedGet) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 3) {
        GTEST_SKIP() << "This test requires at least 3 MPI processes";
    }

    Kokkos::View<double*, Kokkos::HostSpace> data("data", 1);
    data(0) = static_cast<double>(rank);
    
    Window<decltype(data)> window(data, MPI_COMM_WORLD);
    
    if (rank == 2) {
        data(0) = 99.0;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0 || rank == 1) {
        window.lock(Window<decltype(data)>::LockType::Shared, 2);
        
        double value;
        window.get(&value, 1, 2, 0);
        
        window.unlock(2);
        
        EXPECT_EQ(value, 99.0);
    }
}

// Test: Multiple element transfer
TEST(LockUnlockTest, MultipleElements) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        GTEST_SKIP() << "This test requires at least 2 MPI processes";
    }

    const int N = 10;
    Kokkos::View<int*, Kokkos::HostSpace> data("data", N);
    
    for (int i = 0; i < N; i++) {
        data(i) = rank * 100 + i;
    }
    
    Window<decltype(data)> window(data, MPI_COMM_WORLD);
    
    if (rank == 0) {
        window.lock(Window<decltype(data)>::LockType::Exclusive, 1);
        
        int values[N];
        for (int i = 0; i < N; i++) {
            values[i] = i + 1000;
        }
        window.put(values, N, 1, 0);
        
        window.unlock(1);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 1) {
        for (int i = 0; i < N; i++) {
            EXPECT_EQ(data(i), i + 1000);
        }
    }
}

}  // namespace
