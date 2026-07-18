#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <KokkosComm/mpi/window.hpp>

#include <cstdio>
#include <cmath>
#include <algorithm>

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  int rank, size, fails = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size != 2) {
    if (rank == 0) std::fprintf(stderr, "Run with exactly 2 ranks.\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  const int peer = 1 - rank;

  {
    using ViewT     = Kokkos::View<double *, Kokkos::CudaSpace>;
    constexpr int N = 8;
    ViewT v("winbuf", N);

    auto host_mirror_1 = Kokkos::create_mirror_view(v);
    for (int i = 0; i < N; ++i) {
      host_mirror_1(i) = double(rank * N + i);
    }
    Kokkos::deep_copy(v, host_mirror_1);

    KokkosComm::Window<ViewT> win(v, MPI_COMM_WORLD);

    ViewT dst("dst", N);

    // Warmup
    win.fence();
    win.get(dst.data(), N, peer, 0);
    win.fence();

    constexpr int REPS = 50;
    double t_iter[REPS];

    for (int r = 0; r < REPS; ++r) {
      const double t0 = MPI_Wtime();
      win.fence();
      win.get(dst.data(), N, peer, 0);
      win.fence();
      const double t1 = MPI_Wtime();
      t_iter[r]       = t1 - t0;
    }

    std::sort(t_iter, t_iter + REPS);
    printf("[time r%d] transfer min=%.3e median=%.3e s (N=%d, reps=%d)\n", rank, t_iter[0], t_iter[REPS / 2], N, REPS);

    const double tr0 = MPI_Wtime();
    double sum_mine   = 0.0;
    double sum_theirs = 0.0;
    Kokkos::parallel_reduce("sum_mine", N, KOKKOS_LAMBDA(const long i, double &acc) { acc += v(i); }, sum_mine);
    Kokkos::parallel_reduce("sum_theirs", N, KOKKOS_LAMBDA(const long i, double &acc) { acc += dst(i); }, sum_theirs);
    Kokkos::fence();
    const double tr1 = MPI_Wtime();

    const double tr_reduce = tr1 - tr0;

    printf("[time r%d] reduce=%.3e s (N=%d single-shot)\n", rank, tr_reduce, N);

    // Oracle Block:
    const double total    = sum_mine + sum_theirs;
    const double Ntot     = 2.0 * (double)N;
    const double expected = Ntot * (Ntot - 1.0) / 2.0;
    const int f3          = (total == expected) ? 0 : 1;
    printf("[reduce r%d] total=%.1f expected=%.1f -> %s\n", rank, total, expected, f3 ? "FAIL" : "PASS");
    fails += f3;

    // printf("[reduce r%d] sum_mine=%.1f\n", rank, sum_mine);

    int f2 = 0;

    auto host_mirror_3 = Kokkos::create_mirror_view(dst);
    Kokkos::deep_copy(host_mirror_3, dst);

    for (int i = 0; i < N; ++i) {
      if (std::abs(host_mirror_3(i) - double(peer * N + i)) > 1e-14) ++f2;
    }

    printf("[smoke r%d] fence/get: %s\n", rank, f2 ? "FAIL" : "PASS");

    fails += f2;
  }

  int global_fails = 0;

  MPI_Allreduce(&fails, &global_fails, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  if (rank == 0) {
    printf(global_fails ? "SMOKE: FAIL (%d)\n" : "SMOKE: ALL PASS\n", global_fails);
  }

  Kokkos::finalize();
  MPI_Finalize();
  return global_fails ? 1 : 0;
}