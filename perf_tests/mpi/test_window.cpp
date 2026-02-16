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

#include "test_utils.hpp"
#include <KokkosComm/KokkosComm.hpp>
#include <functional>

using Scalar = double;

/*
 * RMA Performance Comparison Benchmarks
 *
 * This file compares different RMA synchronization methods and traditional two-sided communication:
 *
 * 1. Lock/Unlock Put
 *    - Rank 0: Origin (locks rank 1, writes data, unlocks)
 *    - Rank 1: Target (passive - doesn't participate)
 *    - Synchronization: Fine-grained passive target (only 2 processes involved)
 *
 * 2. Lock/Unlock Shared Get
 *    - Rank 0, 1: Origins (both lock rank 2, read data, unlock)
 *    - Rank 2: Target (passive - doesn't participate)
 *    - Synchronization: Shared locks allow concurrent reads
 *
 * 3. Fence Put
 *    - Rank 0: Origin (writes data to rank 1)
 *    - Rank 1: Target (passive)
 *    - Synchronization: Collective (ALL processes must synchronize)
 *
 * 4. PSCW Put (Post-Start-Complete-Wait)
 *    - Rank 0: Origin (starts access, writes to rank 1, completes)
 *    - Rank 1: Target (posts exposure, waits for completion)
 *    - Synchronization: Active target with explicit exposure/access epochs
 *
 * 5. PSCW Get (Post-Start-Complete-Wait)
 *    - Rank 0: Origin (starts access, reads from rank 1, completes)
 *    - Rank 1: Target (posts exposure, waits for completion)
 *    - Synchronization: Active target with explicit exposure/access epochs
 *
 * 6. Send/Recv (TRADITIONAL BASELINE)
 *    - Rank 0: Sender (actively sends)
 *    - Rank 1: Receiver (actively receives)
 *    - Synchronization: Two-sided (both processes participate)
 */

// ============================================================================
// Helper Functions
// ============================================================================

// Write data to rank 1's memory from rank 0 using Lock/Unlock
template <typename Space, typename View>
void lock_unlock_put(benchmark::State &, MPI_Comm comm, const Space &, int rank, const View &v,
                     KokkosComm::Window<View> &window) {
  if (rank == 0) {
    window.lock(KokkosComm::Window<View>::LockType::Exclusive, 1); // Lock rank 1's memory
    window.put(v.data(), v.size(), 1, 0); // Write to rank 1
    window.unlock(1); // Unlock rank 1's memory
  }

  // MPI_Barrier(comm);
}
// Ranks 0 and 1 read data from rank 2's memory using Lock/Unlock Shared
template <typename Space, typename View>
void lock_unlock_shared_get(benchmark::State &, MPI_Comm comm, const Space &, int rank, int size, const View &v,
                            KokkosComm::Window<View> &window) {
  if (size < 3) {
    return; // Skip if not enough processes
  }

  if (rank == 0 || rank == 1) {
    window.lock(KokkosComm::Window<View>::LockType::Shared, 2); // Both lock rank 2's memory
    window.get(v.data(), v.size(), 2, 0); // Both read from rank 2
    window.unlock(2); // Both unlock
  }

  // MPI_Barrier(comm);
}

// Simple fence put from rank 0 to rank 1
template <typename Space, typename View>
void fence_put(benchmark::State &, MPI_Comm, const Space &, int rank, const View &v,
               KokkosComm::Window<View> &window) {
  window.fence(); // All processes sync
  if (rank == 0) {
    window.put(v.data(), v.size(), 1, 0);
  }
  window.fence(); // All processes sync again
}

// Traditional 2-sided send/recv comparison
template <typename Space, typename View>
void sendrecv_comparison(benchmark::State &, MPI_Comm comm, const Space &, int rank, const View &v) {
  if (rank == 0) {
    MPI_Send(v.data(), v.size(), MPI_DOUBLE, 1, 0, comm);
  } else if (rank == 1) {
    MPI_Recv(v.data(), v.size(), MPI_DOUBLE, 0, 0, comm, MPI_STATUS_IGNORE);
  }
}

// PSCW Put from rank 0 to rank 1
template <typename Space, typename View>
void pscw_put(benchmark::State &, MPI_Comm comm, const Space &, int rank, const View &v,
              KokkosComm::Window<View> &window, MPI_Group &origin_group, MPI_Group &target_group) {
  if (rank == 1) {
    // Target: Post window to allow rank 0 to access it
    window.post(origin_group);
  }
  
  if (rank == 0) {
    // Origin: Start access epoch to rank 1's window
    window.start(target_group);
    window.put(v.data(), v.size(), 1, 0);
    window.complete();
  }
  
  if (rank == 1) {
    // Target: Wait for completion
    window.wait();
  }
  
  // MPI_Barrier(comm);
}

// PSCW Get - rank 0 reads from rank 1's memory
template <typename Space, typename View>
void pscw_get(benchmark::State &, MPI_Comm comm, const Space &, int rank, const View &v,
              KokkosComm::Window<View> &window, MPI_Group &origin_group, MPI_Group &target_group) {
  if (rank == 1) {
    // Target: Post window to allow rank 0 to access it
    window.post(origin_group);
  }
  
  if (rank == 0) {
    // Origin: Start access epoch to rank 1's window
    window.start(target_group);
    window.get(v.data(), v.size(), 1, 0);  // GET instead of PUT
    window.complete();
  }
  
  if (rank == 1) {
    // Target: Wait for completion
    window.wait();
  }
  
  // MPI_Barrier(comm);
}

// ============================================================================
// Benchmark Functions
// ============================================================================

void benchmark_lock_unlock_put(benchmark::State &state) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    state.SkipWithError("benchmark_lock_unlock_put needs at least 2 ranks");
    return;
  }

  auto space = Kokkos::DefaultExecutionSpace(); // See test_sendrecv.cpp: 43
  using view_type = Kokkos::View<Scalar *>; // See test_sendrecv.cpp: 44

  const int n = state.range(0);
  view_type v("data", n);
  // See test_osu_latency.cpp: 69

  // Initialize data
  Kokkos::parallel_for("init", n, KOKKOS_LAMBDA(int i) {
    v(i) = static_cast<Scalar>(i);
  });
  Kokkos::fence();

  // Create window once before benchmark loop
  KokkosComm::Window<view_type> window(v, MPI_COMM_WORLD);

  while (state.KeepRunning()) { // See test_osu_latency.cpp: 59
    do_iteration(state, MPI_COMM_WORLD, lock_unlock_put<Kokkos::DefaultExecutionSpace, view_type>,
                 space, rank, v, std::ref(window));
  }

  // See test_sendrecv.cpp: 51
  state.SetBytesProcessed(sizeof(Scalar) * state.iterations() * n);
}

void benchmark_lock_unlock_shared_get(benchmark::State &state) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 3) {
    state.SkipWithError("benchmark_lock_unlock_shared_get needs at least 3 ranks");
    return;
  }

  auto space = Kokkos::DefaultExecutionSpace();
  using view_type = Kokkos::View<Scalar *>;

  const int n = state.range(0);
  view_type v("data", n);

  // Initialize data
  Kokkos::parallel_for("init", n, KOKKOS_LAMBDA(int i) {
    v(i) = static_cast<Scalar>(i);
  });
  Kokkos::fence();

  // Create window once before benchmark loop
  KokkosComm::Window<view_type> window(v, MPI_COMM_WORLD);

  while (state.KeepRunning()) {
    do_iteration(state, MPI_COMM_WORLD, lock_unlock_shared_get<Kokkos::DefaultExecutionSpace, view_type>,
                 space, rank, size, v, std::ref(window));
  }

  state.SetBytesProcessed(sizeof(Scalar) * state.iterations() * n);
}

void benchmark_fence_put(benchmark::State &state) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    state.SkipWithError("benchmark_fence_put needs at least 2 ranks");
    return;
  }

  auto space = Kokkos::DefaultExecutionSpace();
  using view_type = Kokkos::View<Scalar *>;

  const int n = state.range(0);
  view_type v("data", n);

  // Initialize data
  Kokkos::parallel_for("init", n, KOKKOS_LAMBDA(int i) {
    v(i) = static_cast<Scalar>(i);
  });
  Kokkos::fence();

  // Create window once before benchmark loop
  KokkosComm::Window<view_type> window(v, MPI_COMM_WORLD);

  while (state.KeepRunning()) {
    do_iteration(state, MPI_COMM_WORLD, fence_put<Kokkos::DefaultExecutionSpace, view_type>,
                 space, rank, v, std::ref(window));
  }

  state.SetBytesProcessed(sizeof(Scalar) * state.iterations() * n);
}

void benchmark_sendrecv_comparison(benchmark::State &state) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  if (size < 2) {
    state.SkipWithError("benchmark_sendrecv_comparison needs at least 2 ranks");
    return;
  }
  
  auto space = Kokkos::DefaultExecutionSpace();
  using view_type = Kokkos::View<Scalar *>;
  
  const int n = state.range(0);
  view_type v("data", n);
  
  // Initialize data
  Kokkos::parallel_for("init", n, KOKKOS_LAMBDA(int i) {
    v(i) = static_cast<Scalar>(i);
  });
  Kokkos::fence();
  
  while (state.KeepRunning()) {
    do_iteration(state, MPI_COMM_WORLD, sendrecv_comparison<Kokkos::DefaultExecutionSpace, view_type>, 
                 space, rank, v);
  }
  
  state.SetBytesProcessed(sizeof(Scalar) * state.iterations() * n * 2); // send + recv
}

void benchmark_pscw_put(benchmark::State &state) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    state.SkipWithError("benchmark_pscw_put needs at least 2 ranks");
    return;
  }

  auto space = Kokkos::DefaultExecutionSpace();
  using view_type = Kokkos::View<Scalar *>;

  const int n = state.range(0);
  view_type v("data", n);

  // Initialize data
  Kokkos::parallel_for("init", n, KOKKOS_LAMBDA(int i) {
    v(i) = static_cast<Scalar>(i);
  });
  Kokkos::fence();

  // Create window once before benchmark loop
  KokkosComm::Window<view_type> window(v, MPI_COMM_WORLD);
  
  // Create MPI groups for PSCW (once, outside the loop)
  MPI_Group world_group, origin_group, target_group;
  MPI_Comm_group(MPI_COMM_WORLD, &world_group);
  int origin_rank = 0;
  int target_rank = 1;
  MPI_Group_incl(world_group, 1, &origin_rank, &origin_group);
  MPI_Group_incl(world_group, 1, &target_rank, &target_group);

  while (state.KeepRunning()) {
    do_iteration(state, MPI_COMM_WORLD, pscw_put<Kokkos::DefaultExecutionSpace, view_type>,
                 space, rank, v, std::ref(window), std::ref(origin_group), std::ref(target_group));
  }

  // Clean up groups
  MPI_Group_free(&origin_group);
  MPI_Group_free(&target_group);
  MPI_Group_free(&world_group);

  state.SetBytesProcessed(sizeof(Scalar) * state.iterations() * n);
}

void benchmark_pscw_get(benchmark::State &state) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    state.SkipWithError("benchmark_pscw_get needs at least 2 ranks");
    return;
  }

  auto space = Kokkos::DefaultExecutionSpace();
  using view_type = Kokkos::View<Scalar *>;

  const int n = state.range(0);
  view_type v("data", n);

  // Initialize data
  Kokkos::parallel_for("init", n, KOKKOS_LAMBDA(int i) {
    v(i) = static_cast<Scalar>(i);
  });
  Kokkos::fence();

  // Create window once before benchmark loop
  KokkosComm::Window<view_type> window(v, MPI_COMM_WORLD);
  
  // Create MPI groups for PSCW (once, outside the loop)
  MPI_Group world_group, origin_group, target_group;
  MPI_Comm_group(MPI_COMM_WORLD, &world_group);
  int origin_rank = 0;
  int target_rank = 1;
  MPI_Group_incl(world_group, 1, &origin_rank, &origin_group);
  MPI_Group_incl(world_group, 1, &target_rank, &target_group);

  while (state.KeepRunning()) {
    do_iteration(state, MPI_COMM_WORLD, pscw_get<Kokkos::DefaultExecutionSpace, view_type>,
                 space, rank, v, std::ref(window), std::ref(origin_group), std::ref(target_group));
  }

  // Clean up groups
  MPI_Group_free(&origin_group);
  MPI_Group_free(&target_group);
  MPI_Group_free(&world_group);

  state.SetBytesProcessed(sizeof(Scalar) * state.iterations() * n);
}

// ============================================================================
// Benchmark Registration
// ============================================================================

// Test with various message sizes (in number of elements)
BENCHMARK(benchmark_lock_unlock_put)
    ->RangeMultiplier(8)
    ->Range(1, 1<<18)  // 1 to 262,144 elements
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(benchmark_lock_unlock_shared_get)
    ->RangeMultiplier(8)
    ->Range(1, 1<<18)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(benchmark_fence_put)
    ->RangeMultiplier(8)
    ->Range(1, 1<<18)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(benchmark_sendrecv_comparison)
    ->RangeMultiplier(8)
    ->Range(1, 1<<18)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(benchmark_pscw_put)
    ->RangeMultiplier(8)
    ->Range(1, 1<<18)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(benchmark_pscw_get)
    ->RangeMultiplier(8)
    ->Range(1, 1<<18)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);