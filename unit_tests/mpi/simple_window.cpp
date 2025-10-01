#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <KokkosComm/mpi/window.hpp>
#include <iostream>

int main(int argc, char* argv[]) {
	MPI_Init(&argc, &argv);
	Kokkos::initialize(argc, argv);

	Kokkos::finalize();
	MPI_Finalize();

	return 0;
}