#include <gtest/gtest.h>
#include <type_traits>
#include <KokkosComm/KokkosComm.hpp>

namespace {

using namespace KokkosComm::mpi;

// Global variables for benchmarking
int MESSAGE_SIZE = 1024;
int NUM_ITERATIONS = 100;

// Benchmark test fixture
class WindowBenchmark : public testing::Test {
 protected:
	int rank, size;

	void SetUp() override {
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		MPI_Comm_size(MPI_COMM_WORLD, &size);

		if (size < 2) {
			GTEST_SKIP() << "Benchmark needs at least 2 processes";
		}
	}
};

// Benchmark: MPI_Put performance
TEST_F(WindowBenchmark, BenchmarkPut) {

	// Create an array of MESSAGE_SIZE elements
    Kokkos::View<double*, Kokkos::HostSpace> data("data", MESSAGE_SIZE);

	// Initialize data
	if (rank == 0) {
		for (int i = 0; i < MESSAGE_SIZE; i++) {
			data(i) = static_cast<double>(i);
		}
	}

	// Create window
    KokkosComm::Window<decltype(data)> window(data, MPI_COMM_WORLD);


	// Warmup
	for (int i = 0; i < 5; i++) {
		window.fence();
		if (rank == 0) {
			window.put(data.data(), MESSAGE_SIZE, 1, 0);
		}
		window.fence();
	}

	// Actual timed runs
	double total_time = 0.0;

	for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
		// Start timer
		double start_time = MPI_Wtime();
		
		// The operation we're timing
		window.fence();
		if (rank == 0) {
			window.put(data.data(), MESSAGE_SIZE, 1, 0);
		}
		window.fence();
		
		// Stop timer
		double end_time = MPI_Wtime();
		
		// Accumulate time (only rank 0 needs to track)
		if (rank == 0) {
			total_time += (end_time - start_time);
		}
	}

	// Calculate average and print result
	if (rank == 0) {
		double avg_time = total_time / NUM_ITERATIONS;
		
		// Print in CSV format: Operation,MessageSize,Processes,AvgTime
		std::cout << "Put," << MESSAGE_SIZE << "," << size << "," 
				<< avg_time << std::endl;
	}
}

// Benchmark: MPI_Send/Recv performance
TEST_F(WindowBenchmark, BenchmarkSendRecv) {
    // Create array
    Kokkos::View<double*, Kokkos::HostSpace> data("data", MESSAGE_SIZE);
    
    // Initialize
    if (rank == 0) {
        for (int i = 0; i < MESSAGE_SIZE; i++) {
            data(i) = static_cast<double>(i);
        }
    }
    
    // Warmup
    for (int i = 0; i < 5; i++) {
        if (rank == 0) {
            MPI_Send(data.data(), MESSAGE_SIZE, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
        }
        if (rank == 1) {
            MPI_Recv(data.data(), MESSAGE_SIZE, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    
    // Timing
    double total_time = 0.0;
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        double start_time = MPI_Wtime();
        
        if (rank == 0) {
            MPI_Send(data.data(), MESSAGE_SIZE, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
        }
        if (rank == 1) {
            MPI_Recv(data.data(), MESSAGE_SIZE, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        
        double end_time = MPI_Wtime();
        
        if (rank == 0) {
            total_time += (end_time - start_time);
        }
    }
    
    // Print
    if (rank == 0) {
        double avg_time = total_time / NUM_ITERATIONS;
        std::cout << "SendRecv," << MESSAGE_SIZE << "," << size << "," 
                  << avg_time << std::endl;
    }
}
} // Namespace

// Custom main function
int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);
	Kokkos::initialize(argc, argv);

	for (int i = 1; i < argc; i++) {
		std::string arg = argv[i];
		if (arg == "--size" && i + 1 < argc) {
			MESSAGE_SIZE = std::atoi(argv[i+1]);
			i++;
		} else if (arg == "--iterations" && i + 1 < argc) {
			NUM_ITERATIONS = std::atoi(argv[i+1]);
			i++;
		}
	}

	::testing::InitGoogleTest(&argc, argv);
	int result = RUN_ALL_TESTS();

	Kokkos::finalize();
	MPI_Finalize();

	return result;
}