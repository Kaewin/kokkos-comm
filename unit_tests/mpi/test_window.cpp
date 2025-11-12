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
 * 1. Fence-based synchronization
 *    - Basic Put operation (1D_contig_window_put)
 *    - Basic Get operation (1D_contig_window_get)
 *
 * 2. Lock/Unlock synchronization (passive target)
 *    - Exclusive lock Put (LockUnlockPut)
 *    - Exclusive lock Get (LockUnlockGet)
 *    - Shared lock with concurrent Gets (shared_lock_concurrent_get)
 *    - Multi-element array Put (MultiElementLockPut)
 *
 * 3. PSCW synchronization (Post-Start-Complete-Wait)
 *    - PSCW Put operation (PSCWPut)
 *    - PSCW Get operation (PSCWGet)
 *
 * Tested Scalar Types: int, int64_t, float, double, Kokkos::complex<float>, Kokkos::complex<double>
 */

template <typename T>
class WindowTest : public testing::Test {
 public:
  using Scalar = T;
};


// List of types to test - each test will run 6 times, once for each type
using ScalarTypes = ::testing::Types<int, int64_t, float, double, Kokkos::complex<float>, Kokkos::complex<double>>;
TYPED_TEST_SUITE(WindowTest, ScalarTypes);

/*
 * Test: Basic Put Operation with Fence Synchronization
 *
 * Pattern: Fence -> Put -> Fence
 * - Rank 0: Origin (writes value 99 to rank 1)
 * - Rank 1: Target (passive - receives data)
 * - Synchronization: Collective fence (all processes must participate)
 */
template <typename Scalar>
void test_window_put() {
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

TYPED_TEST(WindowTest, 1D_contig_window_put) { test_window_put<typename TestFixture::Scalar>(); }

/*
 * Test: Basic Get Operation with Fence Synchronization
 *
 * Pattern: Fence -> Get -> Fence
 * - Rank 0: Target (passive - provides data)
 * - Rank 1: Origin (reads value from rank 0)
 * - Synchronization: Collective fence (all processes must participate)
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

/*
 * Test: Lock/Unlock Put with Exclusive Lock
 *
 * Pattern: Lock (Exclusive) -> Put -> Unlock
 * - Rank 0: Origin (locks rank 1, writes value 42, unlocks)
 * - Rank 1: Target (passive - doesn't participate in synchronization)
 * - Synchronization: Fine-grained (only 2 processes involved, no collective)
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

TYPED_TEST(WindowTest, LockUnlockPut) { test_lock_unlock_put<typename TestFixture::Scalar>(); }

/*
 * Test: Lock/Unlock Get with Exclusive Lock
 *
 * Pattern: Lock (Exclusive) -> Get -> Unlock
 * - Rank 0: Target (passive - provides data)
 * - Rank 1: Origin (locks rank 0, reads value, unlocks)
 * - Synchronization: Fine-grained (only 2 processes involved, no collective)
 */
template <typename Scalar>
void test_lock_unlock_get() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    GTEST_SKIP() << "This test requires at least 2 MPI processes";
  }

  // Create a single-element view on each process
  Kokkos::View<Scalar*, Kokkos::HostSpace> data("data", 1);
  data(0) = static_cast<Scalar>(rank * 10);

  // Create window exposing local memory for RMA operations
  KokkosComm::Window<decltype(data)> window(data, MPI_COMM_WORLD);

  // Rank 1 locks rank 0's window, gets value, then unlocks
  if (rank == 1) {
    window.lock(KokkosComm::Window<decltype(data)>::LockType::Exclusive, 0);
    Scalar received;
    window.get(&received, 1, 0, 0);
    window.unlock(0);
    // Verify we got rank 0's value (which is 0)
    EXPECT_EQ(received, static_cast<Scalar>(0));
  }

  // Barrier ensures rank 1's operations complete
  MPI_Barrier(MPI_COMM_WORLD);
}

TYPED_TEST(WindowTest, LockUnlockGet) { test_lock_unlock_get<typename TestFixture::Scalar>(); }

/*
 * Test: Shared Lock with Concurrent Get Operations
 *
 * Pattern: Lock (Shared) -> Get -> Unlock (concurrent from multiple origins)
 * - Rank 0: Origin (locks rank 2 with shared lock, reads data)
 * - Rank 1: Origin (locks rank 2 with shared lock, reads data)
 * - Rank 2: Target (passive - provides data to both origins)
 * - Synchronization: Shared locks allow concurrent read access
 */
template <typename Scalar>
void test_window_shared_lock() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 3) {
    GTEST_SKIP() << "This test requires at least 3 MPI processes";
  }

  // Create a single-element view on each process
  Kokkos::View<Scalar*, Kokkos::HostSpace> data("data", 1);
  data(0) = static_cast<Scalar>(rank * 1000);

  // Create window exposing local memory for RMA operations
  KokkosComm::Window<decltype(data)> window(data, MPI_COMM_WORLD);

  // Ranks 0 and 1 both get from rank 2 using shared locks
  if (rank == 0 || rank == 1) {
    window.lock(KokkosComm::Window<decltype(data)>::LockType::Shared, 2);
    Scalar received;
    window.get(&received, 1, 2, 0);
    window.unlock(2);
    // Both should get rank 2's value (2000)
    EXPECT_EQ(received, static_cast<Scalar>(2000));
  }

  // Barrier ensures all operations complete
  MPI_Barrier(MPI_COMM_WORLD);
}

TYPED_TEST(WindowTest, shared_lock_concurrent_get) { test_window_shared_lock<typename TestFixture::Scalar>(); }

/*
 * Test: Multi-Element Array Put with Exclusive Lock
 *
 * Pattern: Lock (Exclusive) -> Put (array) -> Unlock
 * - Rank 0: Origin (locks rank 1, writes 10-element array, unlocks)
 * - Rank 1: Target (passive - receives array)
 * - Synchronization: Fine-grained with exclusive lock
 */
template <typename Scalar>
void test_multi_element_lock_put() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    GTEST_SKIP() << "This test requires at least 2 MPI processes";
  }

  const int n = 10;
  Kokkos::View<Scalar*, Kokkos::HostSpace> data("data", n);

  // Initialize with rank-specific pattern
  for (int i = 0; i < n; ++i) {
    data(i) = static_cast<Scalar>(rank * 100 + i);
  }

  // Create window exposing local memory for RMA operations
  KokkosComm::Window<decltype(data)> window(data, MPI_COMM_WORLD);

  // Rank 0 locks rank 1, puts array, then unlocks
  if (rank == 0) {
    window.lock(KokkosComm::Window<decltype(data)>::LockType::Exclusive, 1);
    
    Kokkos::View<Scalar*, Kokkos::HostSpace> send_data("send", n);
    for (int i = 0; i < n; ++i) {
      send_data(i) = static_cast<Scalar>(i + 1000);
    }
    window.put(send_data.data(), n, 1, 0);
    
    window.unlock(1);
  }

  // Barrier ensures rank 0's operations complete before rank 1 checks
  MPI_Barrier(MPI_COMM_WORLD);

  // Verify all elements were received correctly
  if (rank == 1) {
    for (int i = 0; i < n; ++i) {
      EXPECT_EQ(data(i), static_cast<Scalar>(i + 1000));
    }
  }
}

TYPED_TEST(WindowTest, MultiElementLockPut) { test_multi_element_lock_put<typename TestFixture::Scalar>(); }

/*
 * Test: PSCW put operation
 *
 * Pattern: Post/Start -> Put -> Complete/Wait
 * - Rank 1: Target (posts window to expose memory to rank 0, then waits)
 * - Rank 0: Origin (starts access epoch, writes to rank 1, then completes)
 * - Synchronization: Explicit exposure/access epochs (alternative to fence/lock)
 */
template <typename Scalar>
void test_pscw_put() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    GTEST_SKIP() << "This test requires at least 2 MPI processes";
  }

  // Create a single-element view on each process
  Kokkos::View<Scalar*, Kokkos::HostSpace> data("data", 1);
  data(0) = static_cast<Scalar>(rank);

  // Create window exposing local memory for RMA operations
  KokkosComm::Window<decltype(data)> window(data, MPI_COMM_WORLD);

  // Create MPI groups for PSCW synchronization
  MPI_Group world_group, origin_group, target_group;
  MPI_Comm_group(MPI_COMM_WORLD, &world_group);

  // Rank 0 is origin, rank 1 is target
  int origin_rank = 0;
  int target_rank = 1;
  MPI_Group_incl(world_group, 1, &origin_rank, &origin_group);
  MPI_Group_incl(world_group, 1, &target_rank, &target_group);

  if (rank == 1) {
    // Target: Post window to allow rank 0 to access it
    window.post(origin_group);
  }

  if (rank == 0) {
    // Origin: Start access epoch to rank 1's window
    window.start(target_group);
    
    // Put value 77 to rank 1
    Scalar value = static_cast<Scalar>(77);
    window.put(&value, 1, 1, 0);
    
    // Complete access epoch
    window.complete();
  }

  if (rank == 1) {
    // Target: Wait for all origins to complete their accesses
    window.wait();
    
    // Verify we received the value
    EXPECT_EQ(data(0), static_cast<Scalar>(77));
  }

  // Clean up groups
  MPI_Group_free(&origin_group);
  MPI_Group_free(&target_group);
  MPI_Group_free(&world_group);
}

TYPED_TEST(WindowTest, PSCWPut) { test_pscw_put<typename TestFixture::Scalar>(); }

/*
 * Test: PSCW Get Operation
 *
 * Pattern: Post/Start -> Get -> Complete/Wait
 * - Rank 1: Target (posts window to expose memory to rank 0, then waits)
 * - Rank 0: Origin (starts access epoch, reads from rank 1, then completes)
 * - Synchronization: Explicit exposure/access epochs (alternative to fence/lock)
 */
template <typename Scalar>
void test_pscw_get() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    GTEST_SKIP() << "This test requires at least 2 MPI processes";
  }

  // Create a single-element view on each process
  Kokkos::View<Scalar*, Kokkos::HostSpace> data("data", 1);
  data(0) = static_cast<Scalar>(rank * 100);

  // Create window exposing local memory for RMA operations
  KokkosComm::Window<decltype(data)> window(data, MPI_COMM_WORLD);

  // Create MPI groups for PSCW synchronization
  MPI_Group world_group, origin_group, target_group;
  MPI_Comm_group(MPI_COMM_WORLD, &world_group);

  // Rank 0 is origin, rank 1 is target
  int origin_rank = 0;
  int target_rank = 1;
  MPI_Group_incl(world_group, 1, &origin_rank, &origin_group);
  MPI_Group_incl(world_group, 1, &target_rank, &target_group);

  if (rank == 1) {
    // Target: Post window to allow rank 0 to access it
    window.post(origin_group);
  }

  if (rank == 0) {
    // Origin: Start access epoch to rank 1's window
    window.start(target_group);

    // Get value from rank 1
    Scalar received;
    window.get(&received, 1, 1, 0);

    // Complete access epoch
    window.complete();

    // Verify we received rank 1's value (which is 100)
    EXPECT_EQ(received, static_cast<Scalar>(100));
  }

  if (rank == 1) {
    // Target: Wait for all origins to complete their accesses
    window.wait();
  }

  // Clean up groups
  MPI_Group_free(&origin_group);
  MPI_Group_free(&target_group);
  MPI_Group_free(&world_group);
}

TYPED_TEST(WindowTest, PSCWGet) { test_pscw_get<typename TestFixture::Scalar>(); }

}  // namespace