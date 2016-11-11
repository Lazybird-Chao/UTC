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
		vec_size = num;
		vec1 = (float*)malloc(sizeof(float)*vec_size);
		vec2 = (float*)malloc(sizeof(float)*vec_size);
		vec3 = (float*)malloc(sizeof(float)*vec_size);

		for(int i=0; i<vec_size; i++){
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

		Timer t1;
		t1.start();
		float* vec1_d;
		checkCudaRuntimeErrors(cudaMalloc(&vec1_d, vec_size*sizeof(float)));
		float* vec2_d;
		checkCudaRuntimeErrors(cudaMalloc(&vec2_d, vec_size*sizeof(float)));
		float* vec3_d;
		checkCudaRuntimeErrors(cudaMalloc(&vec3_d, vec_size*sizeof(float)));
		std::cout<<"cuda device space creation time: "<<t1.stop()*1000<<" ms"<<std::endl;

		t1.start();
		checkCudaRuntimeErrors(cudaMemcpy(vec1_d, vec1, vec_size*sizeof(float),cudaMemcpyHostToDevice));
		checkCudaRuntimeErrors(cudaMemcpy(vec2_d, vec2, vec_size*sizeof(float),cudaMemcpyHostToDevice));
		checkCudaRuntimeErrors(cudaMemset(vec3_d, 0, vec_size*sizeof(float)));
		std::cout<<"cuda memcpy in time: "<<t1.stop()*1000<<" ms"<<std::endl;

		/*
		kernelTest<<<1, vec_size/sizeof(float)>>>(vec1_d,vec2_d,vec3_d);
		 */
		GpuKernel mykernel;
		mykernel.setGridDim(vec_size/1024);
		mykernel.setBlockDim(1024);
		mykernel.setNumArgs(3);
		mykernel.setArgs<float*>(0,vec1_d);
		mykernel.setArgs<float*>(1,vec2_d);
		mykernel.setArgs<float*>(2,vec3_d);
		t1.start();
		mykernel.launchKernel((const void*)&kernelTest);
		std::cout<<"cuda kernel run time: "<<t1.stop()*1000<<" ms"<<std::endl;

		t1.start();
		checkCudaRuntimeErrors(cudaMemcpy(vec3, vec3_d, vec_size*sizeof(float),cudaMemcpyDeviceToHost));
		std::cout<<"cuda memcpy out time: "<<t1.stop()*1000<<" ms"<<std::endl;

		float *comp_res = (float*)malloc(sizeof(float)*vec_size);
		t1.start();
		for(int i=0; i<vec_size; i++)
			comp_res[i]= vec1[i]*vec2[i];
		std::cout<<"cpu run time: "<<t1.stop()*1000<<" ms"<<std::endl;
		int err=0;
		/*for(int i=0; i<10; i++){
			std::cout<<vec1[i]<<" "<<vec2[i]<<" "<<vec3[i]<<" "<<comp_res[i]<<std::endl;
		}*/
		for(int i=0; i<vec_size; i++){
			if(comp_res[i] != vec3[i])
				err++;
		}
		if(err>0)
			std::cout<<"error: "<<err<<std::endl;
		else
			std::cout<<"correct!"<<std::endl;

		free(comp_res);
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
	taskA.init(1024*1024);
	taskA.run();
	taskA.wait();
	taskA.finish();


	return 0;

}
