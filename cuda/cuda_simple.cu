#include <cuda.h>
#include <stdio.h>

__host__ void printCudaDevices()
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
//the function which will run on the CUDA device
__global__ void 
cudaIncrement( int *argv ){
	//determine which thread we are
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//increment the idx'th value in the array
	argv[idx]+=idx;
}

int *cudaArray;
int cudaSize;

__host__ void uploadData(int *arr, int size){
	cudaSize=size;
	//allocate an array with the same size on the CUDA device
	cudaMalloc((void**)&cudaArray, sizeof(int)*cudaSize);
	//copy the data from the CPU to the GPU
	cudaMemcpy(cudaArray, arr, sizeof(int)*cudaSize, cudaMemcpyHostToDevice);
}

__host__ void processData(){
	dim3 DimGrid(1,1,1); //num of blocks (x,y,z)
	dim3 DimBlock(cudaSize,1,1); //num of threads
	//invoke the kernel
	cudaIncrement<<<  DimGrid, DimBlock >>>(cudaArray);
}
__host__ void downloadData(int *arr){

	//copy the data from the GPU to the CPU
	cudaMemcpy(arr, cudaArray, sizeof(int)*cudaSize, cudaMemcpyDeviceToHost);
}

//host function, which is called from the CPU
__host__ int
main(int argc, char** argv)
{
	printCudaDevices();
	int arraySize = 100;

	//allocate an array of ints on the cpu
	int *cpuArray=(int*)malloc(arraySize*sizeof(int));
	for(int i=0;i<arraySize;i++) cpuArray[i]=1;

	//print the values of the array
	for(int i=0;i<arraySize;i++) printf("%d ",cpuArray[i]);
 	printf("\n");

	uploadData(cpuArray, arraySize);
	processData();
	downloadData(cpuArray);

	/*
	//allocate an array with the same size on the CUDA device
	int *cudaArray;
	cudaMalloc((void**)&cudaArray, sizeof(int)*arraySize);

	//copy the data from the CPU to the GPU
	cudaMemcpy(cudaArray, cpuArray, sizeof(int)*arraySize, cudaMemcpyHostToDevice);

	dim3 DimGrid(1,1,1); //num of blocks (x,y,z)
	dim3 DimBlock(arraySize,1,1); //num of threads

	//invoke the kernel
	cudaIncrement<<<  DimGrid, DimBlock >>>(cudaArray);

	//copy the data from the GPU to the CPU
	cudaMemcpy(cpuArray, cudaArray, sizeof(int)*arraySize, cudaMemcpyDeviceToHost);

	*/
	//print the values of the array	
	for(int i=0;i<arraySize;i++) printf("%d ",cpuArray[i]);
 	printf("\n");

	//free the arrays
	cudaFree(cudaArray); 
	free(cpuArray); 

	return 0;
}



