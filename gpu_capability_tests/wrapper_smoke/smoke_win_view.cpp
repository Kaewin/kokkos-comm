#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <KokkosComm/mpi/window.hpp>

#include <cstdio>
#include <cmath>

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  int rank, size, fails = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  {
    using ViewT     = Kokkos::View<double *, Kokkos::HostSpace>;
    constexpr int N = 8;
    ViewT v("winbuf", N);

    for (int i = 0; i < N; ++i) {
      v(i) = -1.0;
    }

    KokkosComm::Window<ViewT> win(v, MPI_COMM_WORLD);  // MPI_Win_create under the hood

    // active target
    double src[N];

    for (int i = 0; i < N; ++i) {
      src[i] = 100.0 * rank + i;
    }

    const int right = (rank + 1) % size, left = (rank - 1 + size) % size;

    win.fence();
    win.put(src, N, right, 0);  // 4 arguments: the whole point
    win.fence();

    for (int i = 0; i < N; ++i) {
      if (std::abs(v(i) - (100.0 * left + i)) > 1e-14) ++fails;
    }

    printf("[smoke r%d] fence/put: %s\n", rank, fails ? "FAIL" : "PASS");

    // passive target
    double dst[N];

    win.lock_all();
    win.get(dst, N, left, 0);
    win.flush_all();
    win.unlock_all();

    int f2             = 0;
    const int leftleft = (left - 1 + size) % size;

    for (int i = 0; i < N; ++i) {
      if (std::abs(dst[i] - (100.0 * leftleft + i)) > 1e-14) ++f2;
    }

    printf("[smoke r%d] lock_all/get/flush_all: %s\n", rank, f2 ? "FAIL" : "PASS");

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
