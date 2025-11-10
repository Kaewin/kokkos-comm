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

// ============================================================================
// Test Fixture Setup
// ============================================================================

/*
 * MPI Window Unit Tests
 *
 * This file tests the KokkosComm::Window API for one-sided communication (RMA).
 * Each test is executed for multiple scalar types using Google Test's typed test framework.
 *
 * Test Categories:
 * 1. Fence-based synchronization (put/get operations)
 * 2. Lock-based synchronization (exclusive and shared locks)
 * 3. Multi-element operations
 * 4. Displacement/offset operations
 */

template <typename T>
class WindowTest : public testing::Test {
 public:
  using Scalar = T;
};


// List of types to test - each test will run 6 times, once for each type
using ScalarTypes = ::testing::Types<int, int64_t, float, double, Kokkos::complex<float>, Kokkos::complex<double>>;
TYPED_TEST_SUITE(WindowTest, ScalarTypes);


// ============================================================================
// Test: Basic Put Operation with Fence Synchronization
// ============================================================================

/*
 * Test: Basic Put Operation with Fence Synchronization
 *
 * Pattern: Fence -> Put -> Fence
 * - Rank 0: Origin (writes value 99 to rank 1)
 * - Rank 1: Target (passive - receives data)
 * - Synchronization: Collective fence (all processes must participate)
 *
 * This test validates:
 * - Window creation on a single-element view
 * - MPI_Win_fence synchronization
 * - MPI_Put operation from rank 0 to rank 1
 */
template <typename Scalar>
void test_window() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    GTEST_SKIP() << "This test requires at least 2 MPI processes";
  }

  // Create a single-element view on each process
  Kokkos::View<Scalar*, Kokkos::HostSpace> data("data", 1);
  // Initialize each process's data to its own rank number
  data(0) = static_cast<Scalar>(rank);

  // Create window exposing local memory for RMA operations
  KokkosComm::Window<decltype(data)> window(data, MPI_COMM_WORLD);

  // Begin RMA access epoch
  window.fence();

  // Rank 0 puts value 99 to rank 1's window at displacement 0
  if (rank == 0) {
    Scalar value = static_cast<Scalar>(99);
    window.put(&value, 1, 1, 0);
  }

  // End RMA access epoch and ensure completion
  window.fence();

  // Verify that rank 1 received the value
  if (rank == 1) {
    EXPECT_EQ(data(0), static_cast<Scalar>(99));
  }
}

TYPED_TEST(WindowTest, 1D_contig_window) { test_window<typename TestFixture::Scalar>(); }


// ============================================================================
// Test: Basic Get Operation with Fence Synchronization
// ============================================================================

/*
 * Test: Basic Get Operation with Fence Synchronization
 *
 * Pattern: Fence -> Get -> Fence
 * - Rank 0: Target (passive - provides data)
 * - Rank 1: Origin (reads value from rank 0)
 * - Synchronization: Collective fence (all processes must participate)
 *
 * This test validates:
 * - MPI_Get operation reading from remote rank
 * - Correct data transfer from rank 0 to rank 1's local buffer
 */
template <typename Scalar>
void test_window_get() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    GTEST_SKIP() << "This test requires at least 2 MPI processes";
  }

  // Create a single-element view on each process
  Kokkos::View<Scalar*, Kokkos::HostSpace> data("data", 1);
  // Initialize each process's data to its own rank number
  data(0) = static_cast<Scalar>(rank);

  // Create window exposing local memory for RMA operations
  KokkosComm::Window<decltype(data)> window(data, MPI_COMM_WORLD);

  // Begin RMA access epoch
  window.fence();

  // Rank 1 gets value from rank 0's window
  if (rank == 1) {
    Scalar value = static_cast<Scalar>(-1); // Initialize to sentinel value
    window.get(&value, 1, 0, 0);
    // Verify we got rank 0's value (which is 0)
    EXPECT_EQ(value, static_cast<Scalar>(0));
  }

  // End RMA access epoch and ensure completion
  window.fence();
}

TYPED_TEST(WindowTest, 1D_contig_window_get) { test_window_get<typename TestFixture::Scalar>(); }

// ============================================================================
// Test: Lock/Unlock Put Operation with Exclusive Lock
// ============================================================================

/*
 * Test: Lock/Unlock Put with Exclusive Lock
 *
 * Pattern: Lock (Exclusive) -> Put -> Unlock
 * - Rank 0: Origin (locks rank 1, writes value 42, unlocks)
 * - Rank 1: Target (passive - doesn't participate in synchronization)
 * - Synchronization: Fine-grained (only 2 processes involved, no collective)
 *
 * This test validates:
 * - MPI_Win_lock with exclusive access
 * - MPI_Put operation within lock/unlock epoch
 * - MPI_Win_unlock ensuring operation completion
 */
template <typename Scalar>
void test_lock_unlock_put() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    GTEST_SKIP() << "This test requires at least 2 MPI processes";
  }

  // Create a single-element view on each process
  Kokkos::View<Scalar*, Kokkos::HostSpace> data("data", 1);

  // Initialize each process's data to its own rank number
  data(0) = static_cast<Scalar>(rank);

  // Create window exposing local memory for RMA operations
  KokkosComm::Window<decltype(data)> window(data, MPI_COMM_WORLD);

  // Rank 0 locks rank 1's window exclusively
  if (rank == 0) {
    window.lock(KokkosComm::Window<decltype(data)>::LockType::Exclusive,
                1); // Lock rank 1
    // Put value 42 into rank 1's window
    Scalar value = static_cast<Scalar>(42);
    window.put(&value, 1, 1, 0);
    // Unlock rank 1's window, ensuring completion
    window.unlock(1);
  }

  // Synchronize all processes to ensure rank 1 can check its data
  MPI_Barrier(MPI_COMM_WORLD);

  // Verify that rank 1 received the value
  if (rank == 1) {
    EXPECT_EQ(data(0), static_cast<Scalar>(42));
  }
}

TYPED_TEST(WindowTest, LockUnlockPut) { test_lock_unlock_put<typename TestFixture::Scalar>(); }0

}  // namespace