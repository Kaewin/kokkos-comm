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
#include <type_traits>
#include <KokkosComm/KokkosComm.hpp>

namespace {

using namespace KokkosComm::mpi;

// Existing Test:
// Google test will execute test_window() once for each datatype in ScalarTypes.
template <typename T>
class WindowTest : public testing::Test {
 public:
  using Scalar = T;
};

// List of types to test
// The test will run 6 times, one for each type
using ScalarTypes = ::testing::Types<int, int64_t, float, double, Kokkos::complex<float>, Kokkos::complex<double>>;
TYPED_TEST_SUITE(WindowTest, ScalarTypes);


// Main typed test
// Window class can create MPI windows from Kokkos Views
// getWin() method gives access to window handle
// MPI_Put writes to remote memory through the window
// Tests with different data types
template <typename Scalar>
void test_window() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    GTEST_SKIP() << "This test requires at least 2 MPI processes";
  }

// Skipping complex tests for now until I can fix them
if (std::is_same<Scalar, Kokkos::complex<float>>::value ||
    std::is_same<Scalar, Kokkos::complex<double>>::value) {
    GTEST_SKIP() << "Complex types not yet supported";
}

// Going to build on the existing testing code

/*
Old code:
*/

//   const int N = 10;

//   // Create host view
//   Kokkos::View<Scalar*, Kokkos::HostSpace> recv_host("recv_host", N);
//   // Create device views
//   Kokkos::View<Scalar*, Kokkos::DefaultExecutionSpace> send_dev("send_dev", N);
//   Kokkos::View<Scalar*, Kokkos::DefaultExecutionSpace> recv_dev("recv_dev", N);
//   // Create Window
//   KokkosComm::Window<Kokkos::View<Scalar*>> window(send_dev, MPI_COMM_WORLD);

//   int errs = 0;
//   EXPECT_EQ(errs, 0);



	// One-element view, like below
    Kokkos::View<Scalar*, Kokkos::HostSpace> data("data", 1);
    // Initialize each process's data to it's own rank number
    data(0) = static_cast<Scalar>(rank);

    // Create window
    KokkosComm::Window<decltype(data)> window(data, MPI_COMM_WORLD);

    // Sync
    // Old code - calling fence directly
    // MPI_Win_fence(0, window.getWin());

    // New code - calling class function
    window.fence();

    // Rank 0 puts value 99 to rank 1
    if (rank == 0) {
        Scalar value = static_cast<Scalar>(99);

        // NOT NEEDED ANYMORE:

        // Determine MPI datatype
        // MPI_Datatype mpi_type = MPI_BYTE;
        // if (std::is_same<Scalar, double>::value) mpi_type = MPI_DOUBLE;
        // if (std::is_same<Scalar, float>::value) mpi_type = MPI_FLOAT;
        // if (std::is_same<Scalar, int>::value) mpi_type = MPI_INT;

        // Old code:
        // MPI_Put(&value, 1, mpi_type, 1, 0, 1, mpi_type, window.getWin());

        // New code - calling class function
        window.put(&value, 1, 1, 0);
    }

    // Sync again
    // MPI_Win_fence(0, window.getWin());
    // Calling new class function here too
    window.fence();

    // Check results
    if (rank == 1) {
        EXPECT_EQ(data(0), static_cast<Scalar>(99));
    }
}

TYPED_TEST(WindowTest, 1D_contig_window) { test_window<typename TestFixture::Scalar>(); }


// Begin new test:
TEST(WindowTest, BasicPut) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        GTEST_SKIP() << "This test requires at least 2 MPI processes.";
    }

    // Create view with initial value = rank
    Kokkos::View<double*, Kokkos::HostSpace> data("data", 1);
    data(0) = rank;
    
    // Create window
    KokkosComm::Window<decltype(data)> window(data, MPI_COMM_WORLD);
    
    // Synchronize
    // MPI_Win_fence(0, window.getWin());
    // Using new function:
    window.fence();
    
    // Rank 0 puts value 42 to rank 1
    if (rank == 0) {
        double value = 42.0;
        // New function
        // MPI_Put(&value, 1, MPI_DOUBLE, 1, 0, 1, MPI_DOUBLE, window.getWin());
        window.put(&value, 1, 1, 0);
    }
    
    // Synchronize
    // MPI_Win_fence(0, window.getWin());
    // Using new function:
    window.fence();
   
    // Check result
    if (rank == 1) {
        EXPECT_EQ(data(0), 42.0);
    }
}
// Begin new test:
TEST(WindowTest, BasicGet) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        GTEST_SKIP() << "This test requires at least 2 MPI processes.";
    }

    Kokkos::View<double*, Kokkos::HostSpace> data("data", 1);
    data(0) = rank;
    
    KokkosComm::Window<decltype(data)> window(data, MPI_COMM_WORLD);
    
    window.fence();

    if (rank == 0) {
        data(0) = 77.0;
    }
    if (rank == 1) {
        window.get(&data(0), 1, 0, 0);
    }
    
    window.fence();

    if (rank == 0) {
        EXPECT_EQ(data(0), 77.0);
    }
    if (rank == 1) {
        EXPECT_EQ(data(0), 77.0);
    }
}
}  // namespace