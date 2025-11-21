#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

void checkDeviceMemory(void) {
	size_t free, total;

	cudaMemGetInfo(&free, &total);
	printf("Device memory (free/total) = %lld/%lld bytes\n", free, total);
}

int main(void) {
	int* dDataPtr;
	cudaError_t errorCode;

	checkDeviceMemory();
	errorCode = cudaMalloc(&dDataPtr, sizeof(int) * 1024 * 1024);
	printf("cudaMalloc - %s\n", cudaGetErrorName(errorCode));
	checkDeviceMemory();

	errorCode = cudaMemset(dDataPtr, 0, sizeof(int) * 1024 * 1024);
	printf("cudaMemset - %s\n", cudaGetErrorName(errorCode));

	errorCode = cudaFree(dDataPtr);
	printf("cudaFree - %s\n", cudaGetErrorName(errorCode));
	checkDeviceMemory();
}

// Output:
/*
Device memory (free/total) = 7398752256/8585216000 bytes
cudaMalloc - cudaSuccess
Device memory (free/total) = 7394557952/8585216000 bytes
cudaMemset - cudaSuccess
cudaFree - cudaSuccess
Device memory (free/total) = 7398752256/8585216000 bytes
*/