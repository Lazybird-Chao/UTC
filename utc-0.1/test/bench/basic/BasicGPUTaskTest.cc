/*
 * BasicGPUTaskTest.cc
 *
 *  Created on: Oct 21, 2016
 *      Author: chao
 */

/* main UTC header file */
#include "Utc.h"

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

	void initImpl(){};

	void runImpl(){};


};


/*****************************************************
 * main() program
 ****************************************************/
int main(int argc, char** argv){

	/* initialize UTC context */
	UtcContext &ctx = UtcContext::getContext(&argc, &argv);

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
