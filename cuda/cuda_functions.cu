#include "cuda_functions.h"

/*
* printCudaDevices
*
* Input: None
* Output: None
* Info: get all devices which support cuda, and print them on the console
*/
__host__ void cuda::printCudaDevices()
{
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	for (int device = 0; device < deviceCount; ++device) {
		cudaDeviceProp deviceProp;
		if (cudaGetDeviceProperties(&deviceProp, device) == 0) {
			if (deviceProp.major == 9999 && deviceProp.minor == 9999)
				printf("There is no device supporting CUDA.\n");
			else{
				printf("%d devices found supporting CUDA\n",deviceCount);
				printf("[Device %d -                     %s]\n",device,deviceProp.name);
				printf("[Number of multiprocessors      %d]\n",deviceProp.multiProcessorCount);
				printf("[Running at clockRate           %d]\n",deviceProp.clockRate);
				printf("[Memory avaiable                %d]\n",deviceProp.totalGlobalMem);
				printf("[Shared Memory per Block        %d]\n",deviceProp.sharedMemPerBlock);
				printf("[maximum Threads per Block      %d]\n",deviceProp.maxThreadsPerBlock);
				printf("[maximum Dimension per Block    [%d,%d,%d]]\n",deviceProp.maxThreadsDim[0],deviceProp.maxThreadsDim[1],deviceProp.maxThreadsDim[2]);
				printf("[maximum grid size              [%d,%d,%d]]\n",deviceProp.maxGridSize[0],deviceProp.maxGridSize[1],deviceProp.maxGridSize[2]);
			}
		}
	}
}//printCudaDevices


__host__ void cuda::matrixMulHost(const Matrix A, const Matrix B, Matrix C)
{
	C.width = B.width;
	C.height = A.height;
	for(int row=0;row<A.height;row++){
		for(int col=0;col<B.width;col++){
			*(C.elements + row * C.width + col) = 0.0f;
			for(int colSum=0;colSum<A.width;colSum++){
				*(C.elements + row * C.width + col)+= *(A.elements + row * A.width + colSum)*( *(B.elements + colSum * B.width + col));
			}
		}
	}

}


__host__ void cuda::printMatrix(const Matrix M)
{
	printf("Matrix %dx%d\n",M.height,M.width);
	for(int row=0;row<M.height;row++){
		printf("[");
		for(int col=0;col<M.width;col++){
			printf(" %.2f ",*(M.elements + row * M.width + col));
		}
		printf("]\n");
	}
}

__host__ void cuda::createRandomMatricesHost(int N, int AH, int AW, int BH, int BW, Matrix A[],  Matrix B[], Matrix C[])
{
	for(int k=0;k<N;k++){
		A[k].width = AW;
		A[k].height = AH;
		A[k].elements = (float*) malloc( sizeof(float)*A[k].width*A[k].height );
		fillMatrixRandomValuesHost(A[k]);

		B[k].width = BW;
		B[k].height = BH;
		B[k].elements = (float*) malloc(sizeof(float)*B[k].width*B[k].height);
		fillMatrixRandomValuesHost(B[k]);

		C[k].width = BW;
		C[k].height = AH;
		C[k].elements = (float*) malloc(sizeof(float)*BW*AH);
	}

}
__host__ void cuda::fillMatrixRandomValuesHost(Matrix M)
{
	srand ( time(NULL) );
	for(int row=0;row<M.height;row++){
		for(int col=0;col<M.width;col++){
			*(M.elements + row * M.width + col) = (rand()*RAND_MAX/RAND_MAX)%10+1;
		}
	}
}
__host__ void cuda::performMatrixMultiplicationOnHost(int N, const Matrix A[], const Matrix B[], Matrix C[])
{
	for(int k=0;k<N;k++){
		matrixMulHost(A[k],B[k],C[k]);
	}
}

/*******************************************************************************************/
/*******************************************************************************************/
/*******************************************************************************************/

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)

__global__ void cuda::performMatrixMultiplicationOnDevice(int N, const Matrix A[], const Matrix B[], Matrix C[])
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	cuda::matrixMulDevice(A[idx], B[idx], C[idx]);
	__syncthreads();
}


__device__ void cuda::matrixMulDevice(const Matrix A, const Matrix B, Matrix C)
{
	C.width = B.width;
	C.height = A.height;
	for(int row=0;row<A.height;row++){
		for(int col=0;col<B.width;col++){
			*(C.elements + row * C.width + col) = 0.0;
			for(int colSum=0;colSum<A.width;colSum++){
				*(C.elements + row * C.width + col) += *(A.elements + row * A.width + colSum) * (*(B.elements + colSum * B.width + col));
			}
		}
	}
}

