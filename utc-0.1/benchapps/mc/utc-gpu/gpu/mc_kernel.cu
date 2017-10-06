/*
 * mc_kernel.cu
 *
 *  Created on: Oct 5, 2017
 *      Author: chaoliu
 */

#include "mc_kernel.h"
#include "curand_kernel.h"

__device__ double f(double x){
	return 1.0/(x*x+1);
}

__global__ void mc_kernel(
		double *res,
		double upper,
		double lower,
		unsigned int seed,
		long loop){
	int bx = blockIdx.x;
	int tx = threadIdx.x;

	curandState_t state;
	unsigned long long seq = bx*blockDim.x + tx;
	/* we have to initialize the state */
	curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
	              seq, /* the sequence number should be different for each core (unless you want all
	                             cores to get the same sequence of numbers for some reason - use thread id! */
	              0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
	              &state);
	__shared__ double sum_s[512];
	double sum = 0.0;
	for(long i = 0; i<loop; i++){
		double x = lower + (((double)(curand(&state) % __MAX_RAND__))/__MAX_RAND__)*(upper-lower);
		sum += f(x);
	}
	sum_s[tx] = sum;
	__syncthreads();
	if(tx == 0){
		sum = 0.0;
		for(int i = 0; i<blockDim.x; i++)
			sum += sum_s[i];
		res[bx] = sum;
	}
	__syncthreads();
}



