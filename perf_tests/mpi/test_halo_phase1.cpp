//@HEADER
// Phase 1 halo harness: 2D periodic halo exchange with a SWAPPABLE exchange
// backend. This refactors the existing test_2dhalo.cpp so the communication
// step is isolated behind one function, leaving everything else (decomposition,
// neighbor indexing, halo faces, byte accounting, the square-rank guard) intact.
//
// THIS FILE: two-sided baseline ONLY. RMA backends (Fence/PSCW/Lock-Unlock)
// slot in later, after a correctness check proves the harness is sound.
// OpenMP is intentionally OFF for now -- we validate single-threaded first,
// then enable threading for the Path B crossover experiment.
//
// Structure mirrors test_2dhalo so it drops into the same perf_tests build.
//
// PROVENANCE: Sections A and B below (types, Decomp, the two-sided exchange and
// its benchmark) are taken or directly adapted from the existing KokkosComm
// perf test perf_tests/mpi/test_2dhalo.cpp (Sandia/NTESS, Apache-2.0 WITH
// LLVM-exception). Per-line origin is noted inline. The new contribution begins
// at Section C ("PHASE 2"); everything above it is refactored prior work.
//@HEADER

#include "test_utils.hpp"

#include <iostream>
#include <vector>
#include <cmath>

#include <KokkosComm/KokkosComm.hpp>

// [from test_2dhalo.cpp: identical type aliases, originally local to
//  benchmark_2dhalo(); hoisted to file scope here so all backends share them.]
using Scalar    = double;
using grid_type = Kokkos::View<Scalar ***, Kokkos::LayoutRight>;

// ---------------------------------------------------------------------------
// Decomposition: everything about a rank's place in the periodic rs x rs grid.
// Computed once per benchmark, reused every iteration.
//
// [ADAPTED from test_2dhalo.cpp. The original computed rs/rx/ry inline in
//  benchmark_2dhalo() and the periodic neighbors (xm1/ym1/xp1/yp1) plus the
//  2D->1D get_rank lambda inline in send_recv(). This struct just collects
//  that same arithmetic in one place so it is computed once and reused. The
//  neighbor formulas and the y*rs+x ranking are unchanged from the original;
//  get_rank is renamed to_rank.]
// ---------------------------------------------------------------------------
struct Decomp {
  int rs;                       // ranks per side = floor(sqrt(size))
  int rx, ry;                   // this rank's 2D coordinates
  int xm1, xp1, ym1, yp1;       // neighbor 1D ranks (periodic)
  int nx, ny, nprops;

  static int to_rank(int x, int y, int rs) { return y * rs + x; }

  Decomp(int rank, int size, int nx_, int ny_, int nprops_)
      : nx(nx_), ny(ny_), nprops(nprops_) {
    rs  = static_cast<int>(std::sqrt(static_cast<double>(size)));
    rx  = rank % rs;
    ry  = rank / rs;
    xm1 = to_rank((rx + rs - 1) % rs, ry, rs);
    xp1 = to_rank((rx + 1) % rs,      ry, rs);
    ym1 = to_rank(rx, (ry + rs - 1) % rs, rs);
    yp1 = to_rank(rx, (ry + 1) % rs,      rs);
  }
};

// ---------------------------------------------------------------------------
// Exchange backend: TWO-SIDED baseline. Signature matches what do_iteration
// expects: (state, comm, args...).
//
// [ADAPTED from test_2dhalo.cpp send_recv(). The subview definitions and the
//  send/recv + wait_all sequence are unchanged from the original; the only
//  edits are (a) neighbor ranks now come from the Decomp struct instead of
//  inline get_rank calls, and (b) parameters are passed via Decomp rather than
//  as the original's loose (nx,ny,rx,ry,rs) list. The communication logic is
//  the original's.]
// ---------------------------------------------------------------------------
template <typename Space, typename View>
void exchange_sendrecv(benchmark::State &, MPI_Comm comm, const Space &space,
                       const Decomp &d, const View &v) {
  KokkosComm::Handle<> h{space, comm};
  const int nx = d.nx, ny = d.ny;

  // Boundary faces to send (interior edge) and ghost faces to receive (halo).
  auto xp1_s = Kokkos::subview(v, v.extent(0) - 2, Kokkos::pair{1, ny + 1}, Kokkos::ALL);
  auto xp1_r = Kokkos::subview(v, v.extent(0) - 1, Kokkos::pair{1, ny + 1}, Kokkos::ALL);
  auto xm1_s = Kokkos::subview(v, 1,               Kokkos::pair{1, ny + 1}, Kokkos::ALL);
  auto xm1_r = Kokkos::subview(v, 0,               Kokkos::pair{1, ny + 1}, Kokkos::ALL);
  auto yp1_s = Kokkos::subview(v, Kokkos::pair{1, nx + 1}, v.extent(1) - 2, Kokkos::ALL);
  auto yp1_r = Kokkos::subview(v, Kokkos::pair{1, nx + 1}, v.extent(1) - 1, Kokkos::ALL);
  auto ym1_s = Kokkos::subview(v, Kokkos::pair{1, nx + 1}, 1,               Kokkos::ALL);
  auto ym1_r = Kokkos::subview(v, Kokkos::pair{1, nx + 1}, 0,               Kokkos::ALL);

  std::vector<KokkosComm::Req<>> reqs;
  reqs.push_back(KokkosComm::send(h, xp1_s, d.xp1));
  reqs.push_back(KokkosComm::send(h, xm1_s, d.xm1));
  reqs.push_back(KokkosComm::send(h, yp1_s, d.yp1));
  reqs.push_back(KokkosComm::send(h, ym1_s, d.ym1));
  reqs.push_back(KokkosComm::recv(h, xm1_r, d.xm1));
  reqs.push_back(KokkosComm::recv(h, xp1_r, d.xp1));
  reqs.push_back(KokkosComm::recv(h, ym1_r, d.ym1));
  reqs.push_back(KokkosComm::recv(h, yp1_r, d.yp1));
  KokkosComm::wait_all(reqs);
}

// ---------------------------------------------------------------------------
// Benchmark: two-sided halo.
//
// [ADAPTED from test_2dhalo.cpp benchmark_2dhalo(). The 512x512x3 problem size,
//  the square-rank guard (rank < rs*rs), the idle-remainder branch, the
//  counters, and the SetBytesProcessed accounting are all carried over
//  unchanged. The only change is that the timed exchange is dispatched through
//  the named exchange_sendrecv backend (so other backends can be swapped in),
//  and the idle branch uses an inline no-op lambda in place of the original's
//  free-standing noop().]
// ---------------------------------------------------------------------------
void benchmark_halo_sendrecv(benchmark::State &state) {
  const int nx = 512, ny = 512, nprops = 3;

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  Decomp d(rank, size, nx, ny, nprops);

  if (rank < d.rs * d.rs) {
    auto space = Kokkos::DefaultExecutionSpace();
    grid_type grid("grid", nx + 2, ny + 2, nprops);  // interior + radius-1 halo
    while (state.KeepRunning()) {
      do_iteration(state, MPI_COMM_WORLD,
                   exchange_sendrecv<Kokkos::DefaultExecutionSpace, grid_type>,
                   space, d, grid);
    }
  } else {
    // Non-square remainder ranks idle, exactly as in the original.
    while (state.KeepRunning()) {
      do_iteration(state, MPI_COMM_WORLD, [](benchmark::State &, MPI_Comm) {});
    }
  }

  state.counters["active_ranks"] = d.rs * d.rs;
  state.counters["nx"]           = nx;
  state.counters["nprops"]       = nprops;
  state.SetBytesProcessed(
      sizeof(Scalar) * d.rs * d.rs * state.iterations() * nprops *
      (2 * nx + 2 * nx + 2 * ny + 2 * ny));
}

BENCHMARK(benchmark_halo_sendrecv)->UseManualTime()->Unit(benchmark::kMillisecond);


// ===========================================================================
// PHASE 2: three RMA halo backends (Fence, PSCW, Lock/Unlock) + a correctness
// check that runs ALL THREE against the two-sided result and reports PASS/FAIL
// per mode. They share one face-to-ghost mapping; they differ only in sync.
// Run: --benchmark_filter=benchmark_halo_check    (prints 3 results)
// ===========================================================================

#include <cstdio>

enum HaloDir { HC_XM1 = 0, HC_XP1 = 1, HC_YM1 = 2, HC_YP1 = 3, HC_NDIR = 4 };

using halo_type  = Kokkos::View<Scalar*, Kokkos::LayoutRight>;
using halo_host  = halo_type;   // Serial/HostSpace: mirror == same type

// ---- shared pack / unpack / put -------------------------------------------
void halo_pack(const grid_type& g, halo_type& sbuf, const Decomp& d, int SLOT) {
  // PARALLELIZED: each face cell is independent, so pack it with a Kokkos
  // parallel_for over the OpenMP threads instead of a serial loop. We capture
  // by value [=] so the View handles (g, sbuf -- which are reference-counted,
  // cheap to copy) and the ints are all available inside the thread body.
  const int nx = d.nx, ny = d.ny, np = d.nprops;
  // x-faces: one thread per (j,p) over the y-edge.
  Kokkos::parallel_for("halo_pack_x", ny * np, KOKKOS_LAMBDA(int idx) {
    const int j = idx / np, p = idx % np;
    sbuf(HC_XM1 * SLOT + j * np + p) = g(1,  j + 1, p);
    sbuf(HC_XP1 * SLOT + j * np + p) = g(nx, j + 1, p);
  });
  // y-faces: one thread per (i,p) over the x-edge.
  Kokkos::parallel_for("halo_pack_y", nx * np, KOKKOS_LAMBDA(int idx) {
    const int i = idx / np, p = idx % np;
    sbuf(HC_YM1 * SLOT + i * np + p) = g(i + 1, 1,  p);
    sbuf(HC_YP1 * SLOT + i * np + p) = g(i + 1, ny, p);
  });
  Kokkos::fence();  // make sure both packs finish before we Put the buffer
}

template <typename HaloHost>
void halo_unpack(grid_type& g, const HaloHost& h, const Decomp& d, int SLOT) {
  // PARALLELIZED mirror of halo_pack: scatter received faces into ghost cells,
  // one thread per face cell. h is the received-data View; g is the grid.
  const int nx = d.nx, ny = d.ny, np = d.nprops;
  Kokkos::parallel_for("halo_unpack_x", ny * np, KOKKOS_LAMBDA(int idx) {
    const int j = idx / np, p = idx % np;
    g(0,      j + 1, p) = h(HC_XM1 * SLOT + j * np + p);
    g(nx + 1, j + 1, p) = h(HC_XP1 * SLOT + j * np + p);
  });
  Kokkos::parallel_for("halo_unpack_y", nx * np, KOKKOS_LAMBDA(int idx) {
    const int i = idx / np, p = idx % np;
    g(i + 1, 0,      p) = h(HC_YM1 * SLOT + i * np + p);
    g(i + 1, ny + 1, p) = h(HC_YP1 * SLOT + i * np + p);
  });
  Kokkos::fence();  // ensure unpack completes before the grid is used/checked
}

template <typename Win>
void halo_put_all(Win& win, const halo_type& sbuf, const Decomp& d, int SLOT) {
  // sbuf is now a Kokkos View; .data() gives the raw contiguous host pointer
  // that win.put expects. Offsets index into that flat buffer as before.
  win.put(sbuf.data() + HC_XP1 * SLOT, SLOT, d.xp1, HC_XM1 * SLOT);
  win.put(sbuf.data() + HC_XM1 * SLOT, SLOT, d.xm1, HC_XP1 * SLOT);
  win.put(sbuf.data() + HC_YP1 * SLOT, SLOT, d.yp1, HC_YM1 * SLOT);
  win.put(sbuf.data() + HC_YM1 * SLOT, SLOT, d.ym1, HC_YP1 * SLOT);
}

// Static neighbor group (deduped: small rank counts can repeat a neighbor).
inline MPI_Group make_neighbor_group(const Decomp& d) {
  int cand[4] = {d.xm1, d.xp1, d.ym1, d.yp1};
  int uniq[4]; int n = 0;
  for (int k = 0; k < 4; ++k) {
    bool seen = false;
    for (int m = 0; m < n; ++m) if (uniq[m] == cand[k]) { seen = true; break; }
    if (!seen) uniq[n++] = cand[k];
  }
  MPI_Group wg, ng;
  MPI_Comm_group(MPI_COMM_WORLD, &wg);
  MPI_Group_incl(wg, n, uniq, &ng);
  MPI_Group_free(&wg);
  return ng;
}

// ---- the three backends ----------------------------------------------------
template <typename Win, typename HaloHost>
void exchange_rma_fence(grid_type& g, Win& win, halo_type& hbuf, HaloHost& hh,
                        halo_type& sbuf, const Decomp& d, int SLOT) {
  halo_pack(g, sbuf, d, SLOT);
  win.fence();
  halo_put_all(win, sbuf, d, SLOT);
  win.fence();
  Kokkos::deep_copy(hh, hbuf);
  halo_unpack(g, hh, d, SLOT);
}

template <typename Win, typename HaloHost>
void exchange_rma_pscw(grid_type& g, Win& win, halo_type& hbuf, HaloHost& hh,
                       halo_type& sbuf, const Decomp& d, int SLOT,
                       MPI_Group& nbr) {
  halo_pack(g, sbuf, d, SLOT);
  win.post(nbr);
  win.start(nbr);
  halo_put_all(win, sbuf, d, SLOT);
  win.complete();
  win.wait();
  Kokkos::deep_copy(hh, hbuf);
  halo_unpack(g, hh, d, SLOT);
}

template <typename Win, typename HaloHost>
void exchange_rma_lockunlock(grid_type& g, Win& win, halo_type& hbuf, HaloHost& hh,
                             halo_type& sbuf, const Decomp& d, int SLOT) {
  using LT = typename Win::LockType;
  halo_pack(g, sbuf, d, SLOT);
  win.lock(LT::Shared, d.xp1); win.put(sbuf.data()+HC_XP1*SLOT, SLOT, d.xp1, HC_XM1*SLOT); win.unlock(d.xp1);
  win.lock(LT::Shared, d.xm1); win.put(sbuf.data()+HC_XM1*SLOT, SLOT, d.xm1, HC_XP1*SLOT); win.unlock(d.xm1);
  win.lock(LT::Shared, d.yp1); win.put(sbuf.data()+HC_YP1*SLOT, SLOT, d.yp1, HC_YM1*SLOT); win.unlock(d.yp1);
  win.lock(LT::Shared, d.ym1); win.put(sbuf.data()+HC_YM1*SLOT, SLOT, d.ym1, HC_YP1*SLOT); win.unlock(d.ym1);
  MPI_Barrier(MPI_COMM_WORLD);   // ensure neighbors' puts landed before unpack
  Kokkos::deep_copy(hh, hbuf);
  halo_unpack(g, hh, d, SLOT);
}

// ---- correctness check: run all three, diff each vs two-sided -------------
template <typename Space>
int halo_check_one_mode(const Space& space, const Decomp& d, int mode,
                        const grid_type& g_ref) {
  const int nx = d.nx, ny = d.ny, np = d.nprops;
  const int FACE = (nx > ny ? nx : ny);
  const int SLOT = FACE * np;

  auto init = [&](grid_type g) {
    Kokkos::deep_copy(g, 0.0);
    for (int j = 1; j <= ny; ++j) for (int p = 0; p < np; ++p) {
      g(1,  j, p) = 1000.0*(d.rx+1) + 100.0*(d.ry+1) + j + 0.01*p;
      g(nx, j, p) = 2000.0*(d.rx+1) + 100.0*(d.ry+1) + j + 0.01*p;
    }
    for (int i = 1; i <= nx; ++i) for (int p = 0; p < np; ++p) {
      g(i, 1,  p) = 3000.0*(d.rx+1) + 100.0*(d.ry+1) + i + 0.01*p;
      g(i, ny, p) = 4000.0*(d.rx+1) + 100.0*(d.ry+1) + i + 0.01*p;
    }
  };

  grid_type g("g_mode", nx + 2, ny + 2, np);
  init(g);
  halo_type hbuf("hbuf", HC_NDIR * SLOT);
  Kokkos::deep_copy(hbuf, 0.0);
  auto hh = Kokkos::create_mirror_view(hbuf);
  halo_type sbuf("sbuf", HC_NDIR * SLOT);
  KokkosComm::Window<halo_type> win(hbuf, MPI_COMM_WORLD);

  if (mode == 0) {
    exchange_rma_fence(g, win, hbuf, hh, sbuf, d, SLOT);
  } else if (mode == 1) {
    MPI_Group nbr = make_neighbor_group(d);
    exchange_rma_pscw(g, win, hbuf, hh, sbuf, d, SLOT, nbr);
    MPI_Group_free(&nbr);
  } else {
    exchange_rma_lockunlock(g, win, hbuf, hh, sbuf, d, SLOT);
  }

  int mism = 0;
  auto chk = [&](int i, int j, int p) { if (g_ref(i,j,p) != g(i,j,p)) ++mism; };
  for (int j = 1; j <= ny; ++j) for (int p = 0; p < np; ++p) { chk(0,j,p); chk(nx+1,j,p); }
  for (int i = 1; i <= nx; ++i) for (int p = 0; p < np; ++p) { chk(i,0,p); chk(i,ny+1,p); }
  return mism;
}

// Build the two-sided reference grid once (same pattern as exchange_sendrecv).
template <typename Space>
void build_reference(const Space& space, const Decomp& d, grid_type& g_ref) {
  const int nx = d.nx, ny = d.ny, np = d.nprops;
  Kokkos::deep_copy(g_ref, 0.0);
  for (int j = 1; j <= ny; ++j) for (int p = 0; p < np; ++p) {
    g_ref(1,  j, p) = 1000.0*(d.rx+1) + 100.0*(d.ry+1) + j + 0.01*p;
    g_ref(nx, j, p) = 2000.0*(d.rx+1) + 100.0*(d.ry+1) + j + 0.01*p;
  }
  for (int i = 1; i <= nx; ++i) for (int p = 0; p < np; ++p) {
    g_ref(i, 1,  p) = 3000.0*(d.rx+1) + 100.0*(d.ry+1) + i + 0.01*p;
    g_ref(i, ny, p) = 4000.0*(d.rx+1) + 100.0*(d.ry+1) + i + 0.01*p;
  }
  KokkosComm::Handle<> h{space, MPI_COMM_WORLD};
  auto xp1_s = Kokkos::subview(g_ref, nx,     Kokkos::pair{1, ny+1}, Kokkos::ALL);
  auto xp1_r = Kokkos::subview(g_ref, nx + 1, Kokkos::pair{1, ny+1}, Kokkos::ALL);
  auto xm1_s = Kokkos::subview(g_ref, 1,      Kokkos::pair{1, ny+1}, Kokkos::ALL);
  auto xm1_r = Kokkos::subview(g_ref, 0,      Kokkos::pair{1, ny+1}, Kokkos::ALL);
  auto yp1_s = Kokkos::subview(g_ref, Kokkos::pair{1, nx+1}, ny,     Kokkos::ALL);
  auto yp1_r = Kokkos::subview(g_ref, Kokkos::pair{1, nx+1}, ny + 1, Kokkos::ALL);
  auto ym1_s = Kokkos::subview(g_ref, Kokkos::pair{1, nx+1}, 1,      Kokkos::ALL);
  auto ym1_r = Kokkos::subview(g_ref, Kokkos::pair{1, nx+1}, 0,      Kokkos::ALL);
  std::vector<KokkosComm::Req<>> reqs;
  reqs.push_back(KokkosComm::send(h, xp1_s, d.xp1));
  reqs.push_back(KokkosComm::send(h, xm1_s, d.xm1));
  reqs.push_back(KokkosComm::send(h, yp1_s, d.yp1));
  reqs.push_back(KokkosComm::send(h, ym1_s, d.ym1));
  reqs.push_back(KokkosComm::recv(h, xm1_r, d.xm1));
  reqs.push_back(KokkosComm::recv(h, xp1_r, d.xp1));
  reqs.push_back(KokkosComm::recv(h, ym1_r, d.ym1));
  reqs.push_back(KokkosComm::recv(h, yp1_r, d.yp1));
  KokkosComm::wait_all(reqs);
}

void benchmark_halo_check(benchmark::State& state) {
  const int nx = 16, ny = 16, nprops = 3;
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  Decomp d(rank, size, nx, ny, nprops);
  auto space = Kokkos::DefaultExecutionSpace();
  const char* names[3] = {"Fence", "PSCW", "Lock/Unlock"};

  int total = 0;
  for (auto _ : state) {
    if (rank < d.rs * d.rs) {
      grid_type g_ref("g_ref", nx + 2, ny + 2, nprops);
      build_reference(space, d, g_ref);
      for (int mode = 0; mode < 3; ++mode) {
        int local = halo_check_one_mode(space, d, mode, g_ref);
        int global = 0;
        MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        if (rank == 0) {
          if (global == 0) std::printf("PASS: %-12s ghost cells match two-sided.\n", names[mode]);
          else             std::printf("FAIL: %-12s %d mismatched ghost cells.\n", names[mode], global);
          std::fflush(stdout);
        }
        total += global;
      }
    } else {
      // remainder ranks must still hit the collective ops the active ranks call
      for (int mode = 0; mode < 3; ++mode) {
        int local = 0, global = 0;
        MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
      }
    }
  }
  state.counters["total_mismatches"] = total;
  state.counters["active_ranks"]     = d.rs * d.rs;
}
BENCHMARK(benchmark_halo_check)->UseManualTime()->Iterations(1);

// ===========================================================================
// PHASE 2 timing harness (the H4 measurement): three timed RMA halo
// benchmarks, structured identically to benchmark_halo_sendrecv so all four
// are directly comparable. Window + neighbor group are built ONCE per rank
// (a real solver reuses its window every timestep); only sync + transfer is
// inside the timed loop. Run all four:
//   --benchmark_filter='benchmark_halo_(sendrecv|fence|pscw|lockunlock)'
// ===========================================================================

// Common timed-grid setup mirroring benchmark_halo_sendrecv.
#define HALO_TIMING_PROLOGUE()                                                 \
  const int nx = 512, ny = 512, nprops = 3;                                    \
  int rank, size;                                                              \
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);                                        \
  MPI_Comm_size(MPI_COMM_WORLD, &size);                                        \
  Decomp d(rank, size, nx, ny, nprops);                                        \
  const int FACE = (nx > ny ? nx : ny);                                        \
  const int SLOT = FACE * nprops;

#define HALO_TIMING_EPILOGUE()                                                 \
  state.counters["active_ranks"] = d.rs * d.rs;                                \
  state.counters["nx"]           = nx;                                         \
  state.counters["nprops"]       = nprops;                                     \
  state.SetBytesProcessed(sizeof(Scalar) * d.rs * d.rs * state.iterations() *  \
                          nprops * (2 * nx + 2 * nx + 2 * ny + 2 * ny));

void benchmark_halo_fence(benchmark::State &state) {
  HALO_TIMING_PROLOGUE();
  if (rank < d.rs * d.rs) {
    grid_type grid("grid", nx + 2, ny + 2, nprops);
    halo_type hbuf("hbuf", HC_NDIR * SLOT);
    auto hh = Kokkos::create_mirror_view(hbuf);
    halo_type sbuf("sbuf", HC_NDIR * SLOT);
    KokkosComm::Window<halo_type> win(hbuf, MPI_COMM_WORLD);
    while (state.KeepRunning()) {
      do_iteration(state, MPI_COMM_WORLD,
                   [&](benchmark::State &, MPI_Comm) {
                     exchange_rma_fence(grid, win, hbuf, hh, sbuf, d, SLOT);
                   });
    }
  } else {
    while (state.KeepRunning())
      do_iteration(state, MPI_COMM_WORLD, [](benchmark::State &, MPI_Comm) {});
  }
  HALO_TIMING_EPILOGUE();
}
BENCHMARK(benchmark_halo_fence)->UseManualTime()->Unit(benchmark::kMillisecond);

void benchmark_halo_pscw(benchmark::State &state) {
  HALO_TIMING_PROLOGUE();
  if (rank < d.rs * d.rs) {
    grid_type grid("grid", nx + 2, ny + 2, nprops);
    halo_type hbuf("hbuf", HC_NDIR * SLOT);
    auto hh = Kokkos::create_mirror_view(hbuf);
    halo_type sbuf("sbuf", HC_NDIR * SLOT);
    KokkosComm::Window<halo_type> win(hbuf, MPI_COMM_WORLD);
    MPI_Group nbr = make_neighbor_group(d);   // built ONCE -- the H4 claim
    while (state.KeepRunning()) {
      do_iteration(state, MPI_COMM_WORLD,
                   [&](benchmark::State &, MPI_Comm) {
                     exchange_rma_pscw(grid, win, hbuf, hh, sbuf, d, SLOT, nbr);
                   });
    }
    MPI_Group_free(&nbr);
  } else {
    while (state.KeepRunning())
      do_iteration(state, MPI_COMM_WORLD, [](benchmark::State &, MPI_Comm) {});
  }
  HALO_TIMING_EPILOGUE();
}
BENCHMARK(benchmark_halo_pscw)->UseManualTime()->Unit(benchmark::kMillisecond);

void benchmark_halo_lockunlock(benchmark::State &state) {
  HALO_TIMING_PROLOGUE();
  if (rank < d.rs * d.rs) {
    grid_type grid("grid", nx + 2, ny + 2, nprops);
    halo_type hbuf("hbuf", HC_NDIR * SLOT);
    auto hh = Kokkos::create_mirror_view(hbuf);
    halo_type sbuf("sbuf", HC_NDIR * SLOT);
    KokkosComm::Window<halo_type> win(hbuf, MPI_COMM_WORLD);
    while (state.KeepRunning()) {
      do_iteration(state, MPI_COMM_WORLD,
                   [&](benchmark::State &, MPI_Comm) {
                     exchange_rma_lockunlock(grid, win, hbuf, hh, sbuf, d, SLOT);
                   });
    }
  } else {
    while (state.KeepRunning())
      do_iteration(state, MPI_COMM_WORLD, [](benchmark::State &, MPI_Comm) {});
  }
  HALO_TIMING_EPILOGUE();
}
BENCHMARK(benchmark_halo_lockunlock)->UseManualTime()->Unit(benchmark::kMillisecond);