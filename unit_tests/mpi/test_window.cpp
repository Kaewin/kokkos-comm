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
template <typename T>
class WindowTest : public testing::Test {
 public:
  using Scalar = T;
};

using ScalarTypes = ::testing::Types<int, int64_t, float, double, Kokkos::complex<float>, Kokkos::complex<double>>;
TYPED_TEST_SUITE(WindowTest, ScalarTypes);

template <typename Scalar>
void test_window() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    GTEST_SKIP() << "This test requires at least 2 MPI processes";
  }

  const int N = 10;

  // Create host view
  Kokkos::View<Scalar*, Kokkos::HostSpace> recv_host("recv_host", N);
  // Create device views
  Kokkos::View<Scalar*, Kokkos::DefaultExecutionSpace> send_dev("send_dev", N);
  Kokkos::View<Scalar*, Kokkos::DefaultExecutionSpace> recv_dev("recv_dev", N);
  // Create Window
  KokkosComm::Window<Kokkos::View<Scalar*>> window(send_dev, MPI_COMM_WORLD);

  int errs = 0;
  EXPECT_EQ(errs, 0);
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
    MPI_Win_fence(0, window.getWin());
    
    // Rank 0 puts value 42 to rank 1
    if (rank == 0) {
        double value = 42.0;
        MPI_Put(&value, 1, MPI_DOUBLE, 1, 0, 1, MPI_DOUBLE, window.getWin());
    }
    
    // Synchronize
    MPI_Win_fence(0, window.getWin());
    
    // Check result
    if (rank == 1) {
        EXPECT_EQ(data(0), 42.0);
    }

}


}  // namespace