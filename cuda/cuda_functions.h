#include <cuda.h>
#include <stdio.h>
#include <time.h>

namespace cuda{

	typedef struct {
		int width;
		int height;
		float* elements;
	} Matrix;

	/* all the functions callable on the host */

	__host__ void printCudaDevices();

	__host__ void matrixMulHost(const Matrix A, const Matrix B, Matrix C);

	__host__ void printMatrix(const Matrix M);

	__host__ void fillMatrixRandomValuesHost(Matrix M);

	__host__ void createRandomMatricesHost(int N, int AH, int AW, int BH, int BW, Matrix A[],  Matrix B[], Matrix C[]);

	__host__ void performMatrixMultiplicationOnHost(int N, const Matrix A[], const Matrix B[], Matrix C[]);

	/* all the functions callable from the host on the device */

	__global__ void performMatrixMultiplicationOnDevice(int N, const Matrix A[], const Matrix B[], Matrix C[]);

	/* all the functions callable from the device on the device */

	__device__ void matrixMulDevice(const Matrix A, const Matrix B, Matrix C);

};

