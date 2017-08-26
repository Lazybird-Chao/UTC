/*
 * micro_kernel.cc
 *
 *
 */


#include "micro_kernel.h"


__global__ void micro_kernel(
		float* data,
		int offset,
		int nscale,
		int loop){

	 int i = threadIdx.x + blockIdx.x*blockDim.x;
	 int j = blockIdx.y*blockDim.y + threadIdx.y;
	 int p;
     int t = offset + ((i*(nscale))+j);

     float q = (float)t;
     float s = sinf(q);
     float c = cosf(q);
     data[t] = data[t] + sqrtf(s*s+c*c); //adding 1 to a
     for(p=0;p<loop;p++){
     	q = sinf(q);
       	q = cosf(q);
     	q = sqrtf(s*s+c*c);
     }

}

__global__ void micro_kernel(
		double* data,
		int offset,
		int nscale,
		int loop){

	 int i = threadIdx.x + blockIdx.x*blockDim.x;
	 int j = blockIdx.y*blockDim.y + threadIdx.y;
	 int p;
     int t = offset + ((i*(nscale))+j);

     double q = (float)t;
     double s = sinf(q);
     double c = cosf(q);
     data[t] = data[t] + sqrtf(s*s+c*c); //adding 1 to a
     for(p=0;p<loop;p++){
     	q = sinf(q);
       	q = cosf(q);
     	q = sqrtf(s*s+c*c);
     }

}

