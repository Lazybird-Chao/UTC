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

/* main UTC namespace */
using namespace iUtc;

/*
 * user defined task class
 */
class GPUTaskTest: public UserTaskBase{
public:
	void initImpl(){
		std::cout<<"finish init"<<std::endl;
	};

	void runImpl(){
		std::cout<<"Using utcGPU: "<<getCurrentUtcGpuId()<<std::endl;
		std::cout<<"Mapping to cudaGPU: "<<getCurrentCudaDeviceId()<<std::endl;
		int dev;
		cudaGetDevice(&dev);
		std::cout<<"From cudart: GPU "<<dev<<std::endl;
	};


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
