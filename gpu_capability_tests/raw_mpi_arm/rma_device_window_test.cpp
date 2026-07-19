// rma_device_window_test.cpp
//
// Claim 1, raw-MPI arm (the capability oracle).
// Question under test: can host-initiated MPI RMA (Movie 1) operate on a
// window created over DEVICE-resident memory, on Delta's Cray MPICH?
//
// Anatomy is marked throughout:
//   [ORDER: which processor executes the call]   [HAUL: what carries the bytes]
//
// Usage:   ./rma_dev_test <N elements per rank> [host]
//          second arg "host" = control: identical program, host-resident window.
//
// PASS condition: every rank's sum of ALL 2N elements == Ntot*(Ntot-1)/2
//                 (values are global indices 0..2N-1 — index-sensitive fill,
//                  because all five KRS CGSolve bugs were invisible to bland data).

#include <mpi.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t e_ = (call);                                                   \
    if (e_ != cudaSuccess) {                                                   \
      std::fprintf(stderr, "CUDA error '%s' at %s:%d\n",                       \
                   cudaGetErrorString(e_), __FILE__, __LINE__);                \
      MPI_Abort(MPI_COMM_WORLD, 1);                                            \
    }                                                                          \
  } while (0)

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = -1, size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 2) {
    if (rank == 0) std::fprintf(stderr, "Run with exactly 2 ranks.\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  const long N       = (argc > 1) ? std::atol(argv[1]) : (1L << 20);
  const bool use_dev = !(argc > 2 && std::strcmp(argv[2], "host") == 0);

  // ---- self-documenting header: environment facts printed from inside the
  // ---- run, so no log can ever lie about its own configuration. ------------
  char hname[MPI_MAX_PROCESSOR_NAME]; int hlen = 0;
  MPI_Get_processor_name(hname, &hlen);
  const char* gpuenv = std::getenv("MPICH_GPU_SUPPORT_ENABLED");
  int dev = -1;
  if (use_dev) CUDA_CHECK(cudaGetDevice(&dev));
  std::printf("[rank %d] host=%s mode=%s cuda_dev=%d "
              "MPICH_GPU_SUPPORT_ENABLED=%s N=%ld\n",
              rank, hname, use_dev ? "DEVICE" : "HOST(control)", dev,
              gpuenv ? gpuenv : "(unset)", N);

  // ---- global-index oracle, built on the host --------------------------------
  std::vector<double> h_mine(N), h_theirs(N, -1.0);  // -1 sentinel: a failed Get shows
  for (long i = 0; i < N; ++i) h_mine[i] = double((long)rank * N + i);

  // ---- place "my View" and the landing buffer --------------------------------
  double *mine = nullptr, *theirs = nullptr;
  if (use_dev) {
    CUDA_CHECK(cudaMalloc(&mine,   N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&theirs, N * sizeof(double)));
    // [ORDER: CPU]  [HAUL: DMA engine]  — staging the oracle onto the device
    CUDA_CHECK(cudaMemcpy(mine, h_mine.data(), N * sizeof(double),
                          cudaMemcpyHostToDevice));
  } else {
    mine   = h_mine.data();
    theirs = h_theirs.data();
  }

  // ==== THE LINE UNDER TEST ====================================================
  // Bring-your-own-buffer window over (possibly) device-resident memory.
  // [ORDER: CPU] — punches the deliberate hole in this process's address space.
  MPI_Win win;
  MPI_Win_create(mine, N * sizeof(double), sizeof(double),
                 MPI_INFO_NULL, MPI_COMM_WORLD, &win);
  // ============================================================================

  const int peer = 1 - rank;

  // Warmup
  MPI_Win_fence(0, win);   // open epoch                       [ORDER: CPU]
  // [ORDER: CPU on this rank]  [HAUL: NIC inter-node / GPU-IPC intra-node]
  // Target rank's CPU and GPU cores: idle. That is what "one-sided" means.
  MPI_Get(theirs, N, MPI_DOUBLE, peer, /*target_disp=*/0, N, MPI_DOUBLE, win);
  MPI_Win_fence(0, win);   // close epoch: bytes have landed   [ORDER: CPU]

  constexpr int REPS = 50;
  double t_iter[REPS];

  for(int r = 0; r < REPS; ++r) {
    const double tr0 = MPI_Wtime();
    MPI_Win_fence(0, win);
    MPI_Get(theirs, N, MPI_DOUBLE, peer, 0, N, MPI_DOUBLE, win);
    MPI_Win_fence(0, win);
    const double tr1 = MPI_Wtime();
    t_iter[r] = tr1 - tr0;
  }

  std::sort(t_iter, t_iter + REPS);
  std::printf("[time r%d] transfer min=%.3e median=%.3e s (N=%ld, reps=%d)\n", rank, t_iter[0], t_iter[REPS / 2], N, REPS);

  // ---- verify the TRANSFER (claim 1 is correctness of window + Get) ----------
  if (use_dev) {
    // [ORDER: CPU]  [HAUL: DMA]  — bring landed bytes up for the host-side check
    CUDA_CHECK(cudaMemcpy(h_theirs.data(), theirs, N * sizeof(double),
                          cudaMemcpyDeviceToHost));
  }

  double sum = 0.0;
  for (long i = 0; i < N; ++i) sum += h_mine[i] + h_theirs[i];

  // Values are integers, partial sums stay < 2^53 for N up to ~6e7/rank,
  // so this check is exact — any deviation is a transfer bug, not roundoff.
  const double Ntot     = 2.0 * (double)N;
  const double expected = Ntot * (Ntot - 1.0) / 2.0;
  const double rel      = std::fabs(sum - expected) / expected;
  std::printf("[rank %d] sum=%.10e expected=%.10e rel_err=%.3e -> %s\n",
              rank, sum, expected, rel, (rel < 1e-12) ? "PASS" : "FAIL");

  MPI_Win_free(&win);
  if (use_dev) { CUDA_CHECK(cudaFree(mine)); CUDA_CHECK(cudaFree(theirs)); }
  MPI_Finalize();
  return 0;
}
