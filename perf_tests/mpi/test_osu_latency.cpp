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

// Adapted from the OSU Benchmarks
// Copyright (c) 2002-2024 the Network-Based Computing Laboratory
// (NBCL), The Ohio State University.

#include "test_utils.hpp"

#include <KokkosComm/KokkosComm.hpp>

/// osu_latency_Kokkos_Comm_sendrecv
/// - A small helper that implements a 2-process ping using KokkosComm::send/recv.
/// - Parameters:
///   - benchmark::State& (unused here) : Google benchmark state passed through do_iteration.
///   - MPI_Comm (unused here) : MPI communicator passed through do_iteration for compatibility.
///   - KokkosComm::Handle<>& h : KokkosComm handle used for rank/size and I/O operations.
///   - const View &v : buffer to send / receive.
/// - Behavior:
///   - If this process is rank 0: send `v` to rank 1 using KokkosComm::send and wait for completion.
///   - If rank 1: receive into `v` from rank 0 and wait for completion.
template <typename Space, typename View>
void osu_latency_Kokkos_Comm_sendrecv(benchmark::State &, MPI_Comm, KokkosComm::Handle<> &h, const View &v) {
  if (h.rank() == 0) {
    KokkosComm::wait(KokkosComm::send(h, v, 1));
  } else if (h.rank() == 1) {
    KokkosComm::wait(KokkosComm::recv(h, v, 0));
  }
}

/// benchmark_osu_latency_KokkosComm_sendrecv
/// - Benchmarks the KokkosComm send/recv implementation for latency measurements.
/// - Uses Google Benchmark State to obtain message size via state.range(0).
/// - Creates a Kokkos::View<char*> buffer sized by the benchmark range.
/// - Calls do_iteration to time a single iteration across MPI ranks (manual time via UseManualTime).
/// - Sets a "bytes" counter equal to two messages worth of bytes (send + recv).
void benchmark_osu_latency_KokkosComm_sendrecv(benchmark::State &state) {
  KokkosComm::Handle<> h;
  if (h.size() != 2) {
    state.SkipWithError("benchmark_osu_latency_KokkosComm needs exactly 2 ranks");
  }

  using view_type = Kokkos::View<char *>;
  view_type a("A", state.range(0));

  while (state.KeepRunning()) {
    do_iteration(state, h.mpi_comm(), osu_latency_Kokkos_Comm_sendrecv<Kokkos::DefaultExecutionSpace, view_type>, h, a);
  }
  state.counters["bytes"] = a.size() * 2;
}

/// osu_latency_Kokkos_Comm_mpi_sendrecv
/// - Same ping behavior as osu_latency_Kokkos_Comm_sendrecv but uses the KokkosComm::mpi wrappers
///   that accept an execution space and explicit MPI communicator/rank info.
/// - Parameters:
///   - benchmark::State& (unused) : passed through do_iteration.
///   - MPI_Comm comm : MPI communicator to use for the mpi wrappers.
///   - const Space &space : execution space (e.g., DefaultExecutionSpace).
///   - int rank : rank of this process.
///   - const View &v : buffer to send / receive.
template <typename Space, typename View>
void osu_latency_Kokkos_Comm_mpi_sendrecv(benchmark::State &, MPI_Comm comm, const Space &space, int rank,
                                          const View &v) {
  if (rank == 0) {
    KokkosComm::mpi::send(space, v, 1, 0, comm);
  } else if (rank == 1) {
    KokkosComm::mpi::recv(space, v, 0, 0, comm);
  }
}

/// benchmark_osu_latency_Kokkos_Comm_mpi_sendrecv
/// - Benchmarks the KokkosComm::mpi send/recv wrappers (explicit MPI calls through KokkosComm).
/// - Creates a buffer sized by state.range(0), uses DefaultExecutionSpace, and times iterations
///   via do_iteration which aggregates timing across ranks.
void benchmark_osu_latency_Kokkos_Comm_mpi_sendrecv(benchmark::State &state) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 2) {
    state.SkipWithError("benchmark_osu_latency_KokkosComm needs exactly 2 ranks");
  }

  auto space      = Kokkos::DefaultExecutionSpace();
  using view_type = Kokkos::View<char *>;
  view_type a("A", state.range(0));

  while (state.KeepRunning()) {
    do_iteration(state, MPI_COMM_WORLD, osu_latency_Kokkos_Comm_mpi_sendrecv<Kokkos::DefaultExecutionSpace, view_type>,
                 space, rank, a);
  }
  state.counters["bytes"] = a.size() * 2;
}

/// osu_latency_MPI_isendirecv
/// - Uses raw MPI non-blocking calls (MPI_Irecv / MPI_Isend) to measure latency.
/// - Parameters:
///   - benchmark::State& (unused) : passed through do_iteration.
///   - MPI_Comm comm : communicator for MPI calls.
///   - int rank : process rank.
///   - const View &v : buffer for send/recv.
/// - Behavior:
///   - Rank 0 posts a blocking MPI_Irecv then waits (simulates receive-driven ping).
///   - Rank 1 issues MPI_Isend then waits on the send request.
template <typename View>
void osu_latency_MPI_isendirecv(benchmark::State &, MPI_Comm comm, int rank, const View &v) {
  MPI_Request sendreq, recvreq;
  if (rank == 0) {
    MPI_Irecv(v.data(), v.size(), KokkosComm::Impl::mpi_type<typename View::value_type>(), 1, 0, comm, &recvreq);
    MPI_Wait(&recvreq, MPI_STATUS_IGNORE);
  } else if (rank == 1) {
    MPI_Isend(v.data(), v.size(), KokkosComm::Impl::mpi_type<typename View::value_type>(), 0, 0, comm, &sendreq);
    MPI_Wait(&sendreq, MPI_STATUS_IGNORE);
  }
}

/// benchmark_osu_latency_MPI_isendirecv
/// - Sets up the buffer and benchmarks the previous MPI non-blocking implementation.
/// - Ensures exactly 2 ranks are present, uses state.range(0) for message size,
///   and reports bytes transferred as two messages worth.
void benchmark_osu_latency_MPI_isendirecv(benchmark::State &state) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 2) {
    state.SkipWithError("benchmark_osu_latency_MPI needs exactly 2 ranks");
  }

  using view_type = Kokkos::View<char *>;
  view_type a("A", state.range(0));

  while (state.KeepRunning()) {
    do_iteration(state, MPI_COMM_WORLD, osu_latency_MPI_isendirecv<view_type>, rank, a);
  }
  state.counters["bytes"] = a.size() * 2;
}

/// osu_latency_MPI_sendrecv
/// - Classic blocking MPI_Send / MPI_Recv ping-pong.
/// - Parameters:
///   - benchmark::State& (unused) : passed through do_iteration.
///   - MPI_Comm comm : communicator.
///   - int rank : process rank.
///   - const View &v : buffer to send/receive.
/// - Behavior:
///   - Rank 0 performs MPI_Recv (waits for data from rank 1).
///   - Rank 1 performs MPI_Send (sends data to rank 0).
template <typename View>
void osu_latency_MPI_sendrecv(benchmark::State &, MPI_Comm comm, int rank, const View &v) {
  if (rank == 0) {
    MPI_Recv(v.data(), v.size(), KokkosComm::Impl::mpi_type<typename View::value_type>(), 1, 0, comm,
             MPI_STATUS_IGNORE);
  } else if (rank == 1) {
    MPI_Send(v.data(), v.size(), KokkosComm::Impl::mpi_type<typename View::value_type>(), 0, 0, comm);
  }
}

/// benchmark_osu_latency_MPI_sendrecv
/// - Benchmarks the classic blocking MPI send/recv implementation.
/// - Constructs the view buffer sized by state.range(0) and calls do_iteration to measure latency.
void benchmark_osu_latency_MPI_sendrecv(benchmark::State &state) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 2) {
    state.SkipWithError("benchmark_osu_latency_MPI needs exactly 2 ranks");
  }

  using view_type = Kokkos::View<char *>;
  view_type a("A", state.range(0));

  while (state.KeepRunning()) {
    do_iteration(state, MPI_COMM_WORLD, osu_latency_MPI_sendrecv<view_type>, rank, a);
  }
  state.counters["bytes"] = a.size() * 2;
}

BENCHMARK(benchmark_osu_latency_KokkosComm_sendrecv)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond)
    ->RangeMultiplier(8)
    ->Range(1, 1 << 28);
BENCHMARK(benchmark_osu_latency_Kokkos_Comm_mpi_sendrecv)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond)
    ->RangeMultiplier(8)
    ->Range(1, 1 << 28);
BENCHMARK(benchmark_osu_latency_MPI_isendirecv)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond)
    ->RangeMultiplier(8)
    ->Range(1, 1 << 28);
BENCHMARK(benchmark_osu_latency_MPI_sendrecv)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond)
    ->RangeMultiplier(8)
    ->Range(1, 1 << 28);
