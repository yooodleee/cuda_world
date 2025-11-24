#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// The size of the vector
#define NUM_DATA 1030

// Simple vector sum Kernel
__global__ void vecAdd(int* _a, int* _b, int* _c) {
	int tID = threadIdx.x;
	_c[tID] = _a[tID] + _b[tID];
}

int main(void) {
	int* a, * b, * c, * hc;		// Vectors on the host
	int* da, * db, * dc;		// Vectors on the device

	int memSize = sizeof(int) * NUM_DATA;
	printf("%d elements, memSize = %d bytes\n", NUM_DATA, memSize);

	// Memory allocation on the host-side
	a = new int[NUM_DATA]; memset(a, 0, memSize);
	b = new int[NUM_DATA]; memset(b, 0, memSize);
	c = new int[NUM_DATA]; memset(c, 0, memSize);
	hc = new int[NUM_DATA]; memset(hc, 0, memSize);

	// Data generation
	for (int i = 0; i < NUM_DATA; i++) {
		a[i] = rand() % 10;
		b[i] = rand() % 10;
	}

	// Vector sum on host (for performance comparison)
	for (int i = 0; i < NUM_DATA; i++)
		hc[i] = a[i] + b[i];

	// Memory allocation on the device-side
	cudaMalloc(&da, memSize); cudaMemset(da, 0, memSize);
	cudaMalloc(&db, memSize); cudaMemset(db, 0, memSize);
	cudaMalloc(&dc, memSize); cudaMemset(dc, 0, memSize);

	// Data copy : Host -> Device
	cudaMemcpy(da, a, memSize, cudaMemcpyHostToDevice);
	cudaMemcpy(db, b, memSize, cudaMemcpyHostToDevice);

	// Kernel call
	vecAdd << <ceil(NUM_DATA / 1024), 1024 >> > (da, db, dc);

	// Copy results : Device -> Host
	cudaMemcpy(c, dc, memSize, cudaMemcpyDeviceToHost);

	// Release device memory
	cudaFree(da); cudaFree(db); cudaFree(dc);

	// Check results
	bool result = true;
	for (int i = 0; i < NUM_DATA; i++) {
		if (hc[i] != c[i]) {
			printf("[%d] The result is not matched! (%d, %d)\n", i, hc[i], c[i]);
			result = false;
		}
	}

	if (result)
		printf("GPU works well!\n");

	// Release host memory
	delete[] a; delete[] b; delete[] c;

	return 0;
}

// Output:
/*
1030 elements, memSize = 4120 bytes
[1024] The result is not matched! (11, 0)
[1025] The result is not matched! (3, 0)
[1026] The result is not matched! (13, 0)
[1027] The result is not matched! (12, 0)
[1028] The result is not matched! (13, 0)
[1029] The result is not matched! (13, 0)
*/