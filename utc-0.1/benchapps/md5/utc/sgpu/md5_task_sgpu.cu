/*
 * md5_task_sgpu.cu
 *
 *  Created on: Mar 22, 2017
 *      Author: chao
 */

#include "md5_task_sgpu.h"
#include "md5_kernel.h"
#include "../md5.h"

void MD5SGPU::initImpl(config_t* args){
	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<"begin init ...\n";
		md5Config = args;
	}

	intra_Barrier();
	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<" finish initImpl.\n";
	}
}

void MD5SGPU::runImpl(double* runtime, int blocksize, MemType memtype){
	if(__localThreadId == 0){
		std::cout<<getCurrentTask()->getName()<<" begin run ..."<<std::endl;
	}
	Timer timer;

	/*
	 * create gpumem
	 */
	GpuData<uint8_t> inputs(md5Config->numinputs*md5Config->size, memtype);
	GpuData<uint8_t> out(md5Config->numinputs*DIGEST_SIZE, memtype);

	/*
	 * copy in
	 */
	timer.start();
	inputs.init(md5Config->inputs);
	double copyinTime = timer.stop();

	/*
	 * call kernel
	 */
	if(blocksize > __blocksize)
		blocksize = __blocksize;
	dim3 block(blocksize, 1, 1);
	dim3 grid((md5Config->numinputs + block.x -1)/block.x, 1, 1);
	timer.start();
	for(int i=0; i<md5Config->iterations; i++){
		md5_process<<<grid, block, 0, __streamId>>>(
				inputs.getD(),
				out.getD(true),
				md5Config->numinputs,
				md5Config->size);
		checkCudaErr(cudaGetLastError());
		checkCudaErr(cudaStreamSynchronize(__streamId));
	}
	double kernelTime = timer.stop();

	/*
	 * copy out
	 */
	timer.start();
	out.fetch(md5Config->out);

	runtime[0] = kernelTime + copyinTime + copyoutTime;
	runtime[1] = kernelTime;
	runtime[2] = copyinTime;
	runtime[3] = copyoutTime;
	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<" finish runImpl.\n";
	}

}






