/*
 * BasicGPUTaskTest.cc
 *
 *  Created on: Oct 21, 2016
 *      Author: chao
 */

/* main UTC header file */
#include "Utc.h"
#include "UtcGpu.h"

/* other standard header file */
#include <iostream>
#include <string>
#include <cstdio>
#include <cstring>

#include "BasicGpuTaskTest_kernel.h"


/* main UTC namespace */
using namespace iUtc;

/*
 * user defined task class
 */
class GPUTaskTest: public UserTaskBase{
public:
	void initImpl(int num){
		vec_size = num*sizeof(float);
		vec1 = (float*)malloc(sizeof(float)*vec_size);
		vec2 = (float*)malloc(sizeof(float)*vec_size);
		vec3 = (float*)malloc(sizeof(float)*vec_size);

		for(int i=0; i<vsize; i++){
			vec1[i]= i+1.11;
			vec2[i]= i-2.22;
			vec3[i] = 0;
		}

		std::cout<<"finish init"<<std::endl;

	};

	void runImpl(){
		std::cout<<"Using utcGPU: "<<getCurrentUtcGpuId()<<std::endl;
		std::cout<<"Mapping to cudaGPU: "<<getCurrentCudaDeviceId()<<std::endl;
		int dev;
		cudaGetDevice(&dev);
		std::cout<<"From cudart: GPU "<<dev<<std::endl;

		float* vec1_d;
		vec1_d = cudaMalloc(&vec1_d, vec_size);
		float* vec2_d;
		vec1_d = cudaMalloc(&vec2_d, vec_size);
		float* vec3_d;
		vec1_d = cudaMalloc(&vec3_d, vec_size);

		cudaMemcpy(vec1_d, vec1, vec_size,cudaMemcpyHostToDevice);
		cudaMemcpy(vec2_d, vec2, vec_size,cudaMemcpyHostToDevice);
		cudaMemset(vec3_d, 0, vec_size);

		/*
		kernelTest<<<1, vec_size/sizeof(float)>>>(vec1_d,vec2_d,vec3_d);
		 */
		GpuKernel mykernel;
		mykernel.setGridDim(1);
		mykernel.setBlockDim(vec_size/sizeof(float));
		mykernel.setNumArgs(3);
		mykernel.setArgs<float*>(0,vec1_d);
		mykernel.setArgs<float*>(1,vec2_d);
		mykernel.setArgs<float*>(2,vec3_d);
		mykernel.launchKernel(&kernelTest);

		cudaMemcpy(vec3, vec3_d, vec_size,cudaMemcpyDeviceToHost);

		cudaFree(vec1_d);
		cudaFree(vec2_d);
		cudaFree(vec3_d);

	};

private:
	int vec_size;
	float *vec1;
	float *vec2;
	float *vec3;
};


/*****************************************************
 * main() program
 ****************************************************/
int main(int argc, char** argv){

	/* initialize UTC context */
	UtcContext &ctx = UtcContext::getContext(argc, argv);

	/* get total procs of UTC runtime */
	int nproc = ctx.numProcs();
	/* get current process rank */
	int myProc = ctx.getProcRank();

	/* define task */
	Task<GPUTaskTest> taskA("TaskA", ProcList(0), TaskType::gpu_task);
	taskA.display();

	/* run the task */
	taskA.init();
	taskA.run();
	taskA.wait();
	taskA.finish();


	return 0;

}
