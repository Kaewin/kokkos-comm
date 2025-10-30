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

using Scalar = double;

// ============================================================================
// Helper Functions for Lock/Unlock Operations
// ============================================================================

template <typename Space, typename View>
void lock_unlock_put(benchmark::State &, MPI_Comm comm, const Space &space, int rank, const View &v,
                     KokkosComm::Window<View> &window) {
  if (rank == 0) {
    window.lock(KokkosComm::Window<View>::LockType::Exclusive, 1);
    window.put(v.data(), v.size(), 1, 0);
    window.unlock(1);
  }

  MPI_Barrier(comm);
}

template <typename Space, typename View>
void lock_unlock_shared_get(benchmark::State &, MPI_Comm comm, const Space &space, int rank, int size, const View &v,
                            KokkosComm::Window<View> &window) {
  if (size < 3) {
    return; // Skip if not enough processes
  }

  if (rank == 0 || rank == 1) {
    window.lock(KokkosComm::Window<View>::LockType::Shared, 2);
    window.get(v.data(), v.size(), 2, 0);
    window.unlock(2);
  }

  MPI_Barrier(comm);
}

template <typename Space, typename View>
void fence_put(benchmark::State &, MPI_Comm comm, const Space &space, int rank, const View &v,
               KokkosComm::Window<View> &window) {
  window.fence();
  if (rank == 0) {
    window.put(v.data(), v.size(), 1, 0);
  }
  window.fence();
}

template <typename Space, typename View>
void sendrecv_comparison(benchmark::State &, MPI_Comm comm, const Space &space, int rank, const View &v) {
  if (rank == 0) {
    MPI_Send(v.data(), v.size(), MPI_DOUBLE, 1, 0, comm);
  } else if (rank == 1) {
    MPI_Recv(v.data(), v.size(), MPI_DOUBLE, 0, 0, comm, MPI_STATUS_IGNORE);
  }
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
    do_iteration(state, MPI_COMM_WORLD, lock_unlock_put<Kokkos::DefaultExecutionSpace, view_type>,
                 space, rank, v, window);
  }

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
                 space, rank, size, v, window);
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
                 space, rank, v, window);
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