//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
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

#pragma once

#include <chrono>

# include <mpi.h>
#include <benchmark/benchmark.h>

// F is a function that takes (state, MPI_Comm, args...)
// This function measures the execution time of a single benchmark iteration across all MPI ranks
// Template parameters:
//   F    - Function type to benchmark (must accept state, comm, and optional args)
//   Args - Variable number of additional arguments to pass to the function
template <typename F, typename... Args>
void do_iteration(benchmark::State &state, // Google Benchmark state object (tracks iterations, timing, etc.), the & means pass by reference
                  MPI_Comm comm, // MPI communicator (defines which processes participate)
                  F &&func, // Function to benchmark, the && means it can accept lvalues and rvalues
                  Args... args // Variable number of additional arguments to pass to the function
                ) {
  // Define clock types for high-precision timing measurements
  using Clock    = std::chrono::steady_clock;      // Monotonic clock that never goes backwards
  using Duration = std::chrono::duration<double>;  // Duration type in seconds (double precision)

  // Record start time and execute the function being benchmarked
  auto start = Clock::now();
  func(state, comm, args...);
  Duration elapsed = Clock::now() - start;

  // Synchronize timing across all MPI ranks:
  // We need to find the maximum time taken by any rank since the slowest rank
  // determines the overall iteration time
  double max_elapsed_second;
  double elapsed_seconds = elapsed.count(); // Time taken by this rank in seconds
  MPI_Allreduce(&elapsed_seconds, &max_elapsed_second, 1, MPI_DOUBLE, MPI_MAX, comm);
  /* What MPI_Allreduce does is: 
     It takes the elapsed time from each rank (elapsed_seconds) and computes the maximum value among all ranks.
     This maximum value is stored in max_elapsed_second on all ranks.
     This ensures that when we report the iteration time, we account for the slowest rank, which is critical for accurate performance measurement in parallel applications.
  */
  state.SetIterationTime(max_elapsed_second);  // Report the max time across all ranks, this ensures accurate timing
}
