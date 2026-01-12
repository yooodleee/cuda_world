#include "KernelCall.h"

void main() {
	KernelCall();
	printf("Host code running on CPU\n");
	cudaDeviceSynchronize();
}