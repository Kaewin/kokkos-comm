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

// Global variables for benchmarking
int MESSAGE_SIZE = 1024;
int NUM_ITERATIONS = 100;

// Benchmark test fixture
class LockUnlockBenchmark : public testing::Test {
 protected:
    int rank, size;

    void SetUp() override {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        if (size < 2) {
            GTEST_SKIP() << "Benchmark needs at least 2 processes";
        }
    }
};

// Benchmark: Lock/Unlock Exclusive Put
TEST_F(LockUnlockBenchmark, BenchmarkLockUnlockPut) {
    Kokkos::View<double*, Kokkos::HostSpace> data("data", MESSAGE_SIZE);

    if (rank == 0) {
        for (int i = 0; i < MESSAGE_SIZE; i++) {
            data(i) = static_cast<double>(i);
        }
    }

    Window<decltype(data)> window(data, MPI_COMM_WORLD);

    // Warmup
    for (int i = 0; i < 5; i++) {
        if (rank == 0) {
            window.lock(Window<decltype(data)>::LockType::Exclusive, 1);
            window.put(data.data(), MESSAGE_SIZE, 1, 0);
            window.unlock(1);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Timed runs
    double total_time = 0.0;

    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        double start_time = MPI_Wtime();
        
        if (rank == 0) {
            window.lock(Window<decltype(data)>::LockType::Exclusive, 1);
            window.put(data.data(), MESSAGE_SIZE, 1, 0);
            window.unlock(1);
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        double end_time = MPI_Wtime();
        
        if (rank == 0) {
            total_time += (end_time - start_time);
        }
    }

    if (rank == 0) {
        double avg_time = total_time / NUM_ITERATIONS;
        std::cout << "LockUnlockPut," << MESSAGE_SIZE << "," << size << "," 
                  << avg_time << std::endl;
    }
}

// Benchmark: Fence Put (for comparison)
TEST_F(LockUnlockBenchmark, BenchmarkFencePut) {
    Kokkos::View<double*, Kokkos::HostSpace> data("data", MESSAGE_SIZE);

    if (rank == 0) {
        for (int i = 0; i < MESSAGE_SIZE; i++) {
            data(i) = static_cast<double>(i);
        }
    }

    Window<decltype(data)> window(data, MPI_COMM_WORLD);

    // Warmup
    for (int i = 0; i < 5; i++) {
        window.fence();
        if (rank == 0) {
            window.put(data.data(), MESSAGE_SIZE, 1, 0);
        }
        window.fence();
    }

    // Timed runs
    double total_time = 0.0;

    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        double start_time = MPI_Wtime();
        
        window.fence();
        if (rank == 0) {
            window.put(data.data(), MESSAGE_SIZE, 1, 0);
        }
        window.fence();
        
        double end_time = MPI_Wtime();
        
        if (rank == 0) {
            total_time += (end_time - start_time);
        }
    }

    if (rank == 0) {
        double avg_time = total_time / NUM_ITERATIONS;
        std::cout << "FencePut," << MESSAGE_SIZE << "," << size << "," 
                  << avg_time << std::endl;
    }
}

// Benchmark: Send/Recv (for comparison)
TEST_F(LockUnlockBenchmark, BenchmarkSendRecv) {
    Kokkos::View<double*, Kokkos::HostSpace> data("data", MESSAGE_SIZE);
    
    if (rank == 0) {
        for (int i = 0; i < MESSAGE_SIZE; i++) {
            data(i) = static_cast<double>(i);
        }
    }
    
    // Warmup
    for (int i = 0; i < 5; i++) {
        if (rank == 0) {
            MPI_Send(data.data(), MESSAGE_SIZE, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
        }
        if (rank == 1) {
            MPI_Recv(data.data(), MESSAGE_SIZE, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    
    // Timed runs
    double total_time = 0.0;
    
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        double start_time = MPI_Wtime();
        
        if (rank == 0) {
            MPI_Send(data.data(), MESSAGE_SIZE, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
        }
        if (rank == 1) {
            MPI_Recv(data.data(), MESSAGE_SIZE, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        
        double end_time = MPI_Wtime();
        
        if (rank == 0) {
            total_time += (end_time - start_time);
        }
    }
    
    if (rank == 0) {
        double avg_time = total_time / NUM_ITERATIONS;
        std::cout << "SendRecv," << MESSAGE_SIZE << "," << size << "," 
                  << avg_time << std::endl;
    }
}

// Benchmark: Lock/Unlock Shared Get (if 3+ processes)
TEST_F(LockUnlockBenchmark, BenchmarkLockUnlockSharedGet) {
    if (size < 3) {
        GTEST_SKIP() << "This benchmark requires at least 3 processes";
    }

    Kokkos::View<double*, Kokkos::HostSpace> data("data", MESSAGE_SIZE);

    for (int i = 0; i < MESSAGE_SIZE; i++) {
        data(i) = static_cast<double>(i);
    }

    Window<decltype(data)> window(data, MPI_COMM_WORLD);

    // Warmup
    for (int i = 0; i < 5; i++) {
        if (rank == 0 || rank == 1) {
            window.lock(Window<decltype(data)>::LockType::Shared, 2);
            window.get(data.data(), MESSAGE_SIZE, 2, 0);
            window.unlock(2);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Timed runs
    double total_time = 0.0;

    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        double start_time = MPI_Wtime();
        
        if (rank == 0 || rank == 1) {
            window.lock(Window<decltype(data)>::LockType::Shared, 2);
            window.get(data.data(), MESSAGE_SIZE, 2, 0);
            window.unlock(2);
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        double end_time = MPI_Wtime();
        
        if (rank == 0) {
            total_time += (end_time - start_time);
        }
    }

    if (rank == 0) {
        double avg_time = total_time / NUM_ITERATIONS;
        std::cout << "LockUnlockSharedGet," << MESSAGE_SIZE << "," << size << "," 
                  << avg_time << std::endl;
    }
}

}  // namespace

// Custom main function for command-line args
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);

    // Parse command-line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--size" && i + 1 < argc) {
            MESSAGE_SIZE = std::atoi(argv[i+1]);
            i++;
        } else if (arg == "--iterations" && i + 1 < argc) {
            NUM_ITERATIONS = std::atoi(argv[i+1]);
            i++;
        }
    }

    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

    Kokkos::finalize();
    MPI_Finalize();

    return result;
}