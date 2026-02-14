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

template <typename T>
class WindowTest : public testing::Test {
 public:
  using Scalar = T;
};

// Kokkos::complex<float> excluded: MPI_Accumulate + MPI_SUM fails with MPI_COMPLEX (Fortran type) in Open MPI
// using ScalarTypes = ::testing::Types<int, int64_t, float, double, Kokkos::complex<float>, Kokkos::complex<double>>;
using ScalarTypes = ::testing::Types<int, int64_t, float, double, Kokkos::complex<double>>;
TYPED_TEST_SUITE(WindowTest, ScalarTypes);

template <typename Scalar>
void test_fence_put() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    GTEST_SKIP() << "This test requires at least 2 MPI processes";
  }

  Kokkos::View<Scalar*, Kokkos::HostSpace> data("data", 1);
  data(0) = static_cast<Scalar>(rank);

  KokkosComm::Window<decltype(data)> window(data, MPI_COMM_WORLD);

  window.fence();

  if (rank == 0) {
    Scalar value = static_cast<Scalar>(99);
    window.put(&value, 1, 1, 0);
  }

  window.fence();

  if (rank == 1) {
    EXPECT_EQ(data(0), static_cast<Scalar>(99));
  }
}

TYPED_TEST(WindowTest, FencePut) { test_fence_put<typename TestFixture::Scalar>(); }

template <typename Scalar>
void test_fence_get() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    GTEST_SKIP() << "This test requires at least 2 MPI processes";
  }

  Kokkos::View<Scalar*, Kokkos::HostSpace> data("data", 1);
  data(0) = static_cast<Scalar>(rank);

  KokkosComm::Window<decltype(data)> window(data, MPI_COMM_WORLD);

  window.fence();

  if (rank == 1) {
    Scalar value = static_cast<Scalar>(-1); 
    window.get(&value, 1, 0, 0);
    EXPECT_EQ(value, static_cast<Scalar>(0));
  }

  window.fence();
}

TYPED_TEST(WindowTest, FenceGet) { test_fence_get<typename TestFixture::Scalar>(); }

template <typename Scalar>
void test_fence_accumulate() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    GTEST_SKIP() << "This test requires at least 2 MPI processes";
  }

  Kokkos::View<Scalar*, Kokkos::HostSpace> data("data", 1);
  data(0) = static_cast<Scalar>(rank);

  KokkosComm::Window<decltype(data)> window(data, MPI_COMM_WORLD);

  window.fence();

  if (rank == 0) {
    Scalar value = static_cast<Scalar>(10);
    window.accumulate(&value, 1, 1, 0, MPI_SUM);
  }

  window.fence();

  if (rank == 1) {
    EXPECT_EQ(data(0), static_cast<Scalar>(11));
  }
}

TYPED_TEST(WindowTest, FenceAccumulate) { test_fence_accumulate<typename TestFixture::Scalar>();}

template <typename Scalar>
void test_lock_unlock_exclusive_put() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    GTEST_SKIP() << "This test requires at least 2 MPI processes";
  }

  Kokkos::View<Scalar*, Kokkos::HostSpace> data("data", 1);
  data(0) = static_cast<Scalar>(rank);

  KokkosComm::Window<decltype(data)> window(data, MPI_COMM_WORLD);

  if (rank == 0) {
    window.lock(KokkosComm::Window<decltype(data)>::LockType::Exclusive, 1);
    Scalar value = static_cast<Scalar>(0);
    window.put(&value, 1, 1, 0);
    window.unlock(1);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 1) {
    EXPECT_EQ(data(0), static_cast<Scalar>(0));
  }
}

TYPED_TEST(WindowTest, LockUnlockExclusivePut) { test_lock_unlock_exclusive_put<typename TestFixture::Scalar>();}

template <typename Scalar>
void test_lock_unlock_exclusive_get() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    GTEST_SKIP() << "This test requires at least 2 MPI processes";
  }

  Kokkos::View<Scalar*, Kokkos::HostSpace> data("data", 1);
  data(0) = static_cast<Scalar>(rank * 100);

  KokkosComm::Window<decltype(data)> window(data, MPI_COMM_WORLD);

  if (rank == 1) {
    window.lock(KokkosComm::Window<decltype(data)>::LockType::Exclusive, 0);
    Scalar value = static_cast<Scalar>(-1);
    window.get(&value, 1, 0, 0);
    window.unlock(0);
    EXPECT_EQ(value, static_cast<Scalar>(0));
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) {
    EXPECT_EQ(data(0), static_cast<Scalar>(0));
  }
}

TYPED_TEST(WindowTest, LockUnlockExclusiveGet) { test_lock_unlock_exclusive_get<typename TestFixture::Scalar>(); }

template <typename Scalar>
void test_lock_unlock_exclusive_accumulate() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    GTEST_SKIP() << "This test requires at least 2 MPI processes";
  }

  Kokkos::View<Scalar*, Kokkos::HostSpace> data("data", 1);
  data(0) = static_cast<Scalar>(rank);

  KokkosComm::Window<decltype(data)> window(data, MPI_COMM_WORLD);

  if (rank == 0) {
    Scalar value = static_cast<Scalar>(10);
    window.accumulate(&value, 1, 1, 0, MPI_SUM);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 1) {
    EXPECT_EQ(data(0), static_cast<Scalar>(11));
  }

}

TYPED_TEST(WindowTest, LockUnlockExclusiveAccumulate) { test_lock_unlock_exclusive_accumulate<typename TestFixture::Scalar>(); }

}  // namespace