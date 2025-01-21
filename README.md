# mpiPlusOpenMP

Simple program file that demonstrates how to initialize MPI and do a matrix multiplication on rank 0 but spawn threads via openMP. 

If compiling using intel,
mpic++ -fpic -qopenmp -o myprog.exe matrixMultiply_MPIplusOpenMP.cpp
