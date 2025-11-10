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

template <typename Scalar>
void test_window() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    GTEST_SKIP() << "This test requires at least 2 MPI processes";
  }

	// One-element view, like below
    Kokkos::View<Scalar*, Kokkos::HostSpace> data("data", 1);
    // Initialize each process's data to its own rank number
    data(0) = static_cast<Scalar>(rank);

    // Create window
    KokkosComm::Window<decltype(data)> window(data, MPI_COMM_WORLD);

    window.fence();

    // Rank 0 puts value 99 to rank 1
    if (rank == 0) {
        Scalar value = static_cast<Scalar>(99);
        window.put(&value, 1, 1, 0);
    }

    window.fence();

    // Check results
    if (rank == 1) {
        EXPECT_EQ(data(0), static_cast<Scalar>(99));
    }
}

TYPED_TEST(WindowTest, 1D_contig_window) { test_window<typename TestFixture::Scalar>(); }

}  // namespace