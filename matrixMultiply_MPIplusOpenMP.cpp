#include <iostream>
#include <omp.h>
#include <mpi.h>
#include <cstdlib> // for std::atoi
#include <vector>

int main(int argc, char* argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get the rank of the current process
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Get the total number of processes
    int ncpus;
    MPI_Comm_size(MPI_COMM_WORLD, &ncpus);

    // Print a message from each process
    std::cout << "Hello, world! from process " << rank << " of " << ncpus << std::endl;

    // Set some defaults
    int numThreads = 1; // Default to 1 thread
	int N = 1000; // Default size of the matrices

	// Check for arguments specifying thread count and matrix size
    if (argc > 1) {
		numThreads = std::atoi(argv[1]); // Convert argument to integer
    } 
	if (argc > 2) {
		N = std::atoi(argv[2]); // Convert argument to integer
	}

	if(rank == 0) {
		std::cout << "Matrix size set to " << N << std::endl;
		// Output the number of processes and threads requested
	    std::cout << "Running the application using " << ncpus << " processes "
		          << "and " << numThreads << " threads." << std::endl;
	}

    std::vector<std::vector<int>> A(N, std::vector<int>(N));
    std::vector<std::vector<int>> B(N, std::vector<int>(N));
    std::vector<std::vector<int>> C(N, std::vector<int>(N, 0));
	 
	// Set the number of threads
	omp_set_num_threads(numThreads);

    // Initialize matrices A and B
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i][j] = i + j;
            B[i][j] = i - j;
        }
    }

    // Start the timer
    double start_time = omp_get_wtime();

    // Perform matrix multiplication C = A * B
	if(rank == 0) {
		#pragma omp parallel for
		for (int i = 0; i < N; ++i) {
			for (int j = 0; j < N; ++j) {
				int sum = 0;
				for (int k = 0; k < N; ++k) {
					sum += A[i][k] * B[k][j];
				}
                C[i][j] = sum;
            }
        }		
	}

    // Stop the timer
    double end_time = omp_get_wtime();

    // Calculate elapsed time
    double elapsed_time = end_time - start_time;

	if(rank == 0) {
		std::cout << "Matrix multiplication completed." << std::endl;
		std::cout << "Elapsed Time: " << elapsed_time << " seconds" << std::endl;
	}
	
    int diagonalSum = 0;
	int lastRowSum = 0;
	if (rank == 0) {
	  for (int i = 0; i < N; ++i) {
        diagonalSum += C[i][i];
      }	  	
	  for (int j = 0; j < N; ++j) {
		lastRowSum += C[N - 1][j]; // Accessing the last row (index N-1)
      }
	  
	  std::cout << "Matrix diagonal sum: " << diagonalSum << std::endl;		
	  std::cout << "Matrix lastRowSum: " << lastRowSum << std::endl;			  
	}
	
	// Require the barrier here as we don't want other processes to finalize
	// MPI before rank 0 has completed the work of matrix multiplication.	
	MPI_Barrier(MPI_COMM_WORLD);

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
