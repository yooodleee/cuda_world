#include "KernelCall.h"

__global__ void kernel(void) {
	printf("Device code running on the GPU\n");
}

void KernelCall(void) {
	kernel << <1, 10 >> > ();
}