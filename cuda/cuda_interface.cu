#include <cuda.h>
#include <stdio.h>
#include "cuda_functions.h"
//matrix in shared memory
//extern __shared__ float *matrix;

__device__ void showSharedMemory(float *sM);
__global__ void matrixInverse(float *m);
__global__ void matrixMul(float *matrix, int *thread, int *blockid, int *blockdim);
__global__ void matrixSVD(float *A, float *U, float *S, float *V);
__global__ void getKernelMemory(float *m);
__global__ void showKernelSize(int *m);

// Host function
int
main(int argc, char** argv)
{
	cuda::printCudaDevices();


	//multiplying a MxN and a NxM matrix
	const int Nmults =512;
	const int AH = 512;
	const int AW = 512;
	const int BH = AW;
	const int BW = AH;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	cuda::Matrix A[Nmults];
	cuda::Matrix B[Nmults];
	cuda::Matrix C[Nmults];
	cuda::createRandomMatricesHost(Nmults, AH, AW, BH, BW, A,  B, C);

	printf("Start of %d [%dx%d]*[%dx%d] matrix multiplications....\n",Nmults, AH,AW,BH,BW);
	cuda::performMatrixMultiplicationOnHost(Nmults, A, B, C);
	printf("Done.\n");

	//cuda::printMatrix(A[0]);
	//cuda::printMatrix(B[0]);
	//cuda::printMatrix(C[0]);

	/*** record the time ***/
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime = 0.0f;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("\nelapsed time multiplication on Host: %.4fs\n", elapsedTime/1000);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	/***********************/

	//cuda::printMatrix(C[0]);

	cudaEvent_t startDevice, stopDevice;
	cudaEventCreate(&startDevice);
	cudaEventCreate(&stopDevice);
	cudaEventRecord(startDevice, 0);

	//int sizeM = sizeof(cuda::Matrix)*Nmults;

	int sizeA = AH*AW*sizeof(float);
	int sizeB = BH*BW*sizeof(float);
	int sizeC = BW*AH*sizeof(float);

	cuda::Matrix Ad[Nmults];
	cuda::Matrix Bd[Nmults];
	cuda::Matrix Cd[Nmults];
	cuda::Matrix COut[Nmults];

	for(int i=0;i<Nmults;i++){
		Ad[i].width = AW;
		Ad[i].height = AH;
		Bd[i].width = BW;
		Bd[i].height = BH;
		Cd[i].width = BW;
		Cd[i].height = AH;
		cudaMalloc((void**)&Ad[i].elements, sizeA);
		cudaMalloc((void**)&Bd[i].elements, sizeB);
		cudaMalloc((void**)&Cd[i].elements, sizeC);
		cudaMemcpy(Ad[i].elements, A[i].elements, sizeA, cudaMemcpyHostToDevice);
		cudaMemcpy(Bd[i].elements, B[i].elements, sizeB, cudaMemcpyHostToDevice);
		cudaMemcpy(Cd[i].elements, C[i].elements, sizeC, cudaMemcpyHostToDevice);
		COut[i].width = BW;
		COut[i].height = AH;
		COut[i].elements = (float *)malloc(sizeC);
	}

	// set the grid and block sizes
	dim3   DimGrid(1,1,1); //num of blocks (x,y,z)
	dim3   DimBlock(Nmults,1,1); //num of threads
	size_t SharedMemBytes = 1; //bytes of shared memory

	for(int i=0;i<Nmults;i++){
		cudaMemcpy(COut[i].elements, Cd[i].elements, sizeC, cudaMemcpyDeviceToHost);
	}

	//cuda::printMatrix(COut[0]);

	// invoke the kernel
	printf("\nStart multiplication on Device....\n");
	cuda::performMatrixMultiplicationOnDevice<<< DimGrid, DimBlock, SharedMemBytes >>>(Nmults, Ad, Bd, Cd);
	printf("Done.\n");

	for(int i=0;i<Nmults;i++){
		cudaMemcpy(COut[i].elements, Cd[i].elements, sizeC, cudaMemcpyDeviceToHost);
	}

	float error = 0.0f;
	for(int i=0;i<Nmults;i++){
		for(int row=0;row<C[i].height;row++){
			float sumUp = 0.0f;
			for(int col=0;col<C[i].width;col++){
				sumUp += abs(*(C[i].elements + row * C[i].width + col)-*(COut[i].elements + row * COut[i].width + col));
			}
			if(sumUp>0.1) printf("matrix no. %d was not correct calculated..\n",i);
			error+=sumUp;
		}
	}
	printf("difference error = %.4f\n",error);
	
	//cuda::printMatrix(COut[0]);

	/*** record the time ***/
	cudaEventRecord(stopDevice, 0);
	cudaEventSynchronize(stopDevice);
	float elapsedTimeDevice;
	cudaEventElapsedTime(&elapsedTimeDevice, startDevice, stopDevice);
	printf("\nelapsed time multiplication on Device (with copy): %.4fs\n", elapsedTimeDevice/1000);
	cudaEventDestroy(startDevice);
	cudaEventDestroy(stopDevice);
	/***********************/


	/* free everything */
	for(int i=0;i<Nmults;i++){
		cudaFree(Ad[i].elements);
		cudaFree(Bd[i].elements);
		cudaFree(Cd[i].elements);
		free(A[i].elements);
		free(B[i].elements);
		free(C[i].elements);
		free(COut[i].elements);
	}
	return 0;
}

__global__ void
matrixMul(float *matrix, int *thread, int *blockid, int *blockdim)
{
	// determine where in the thread grid we are
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	float icl = clock();
	//int idtransx = blockIdx.x * blockDim.x + threadIdx.x;
	__prof_trigger(0);
	matrix[idx] = icl;
	thread[idx] = threadIdx.x;
	blockid[idx] = blockIdx.x;
	blockdim[idx] = gridDim.z;
	__syncthreads();
}


__global__ void
showKernelSize(int *m)
{
	m[0] = gridDim.x;
	m[1] = blockDim.x;
	m[2] = blockIdx.x;
	m[4] = threadIdx.x;

	m[5] = gridDim.y;
	m[6] = blockDim.y;
	m[7] = blockIdx.y;
	m[8] = threadIdx.y;

	m[9] = gridDim.z;
	m[10] = blockDim.z;
	m[11] = blockIdx.z;
	m[12] = threadIdx.z;
}

__device__ void showSharedMemory(float *sM){
	//int idx = blockIdx.x * blockDim.x + threadIdx.x;

}
