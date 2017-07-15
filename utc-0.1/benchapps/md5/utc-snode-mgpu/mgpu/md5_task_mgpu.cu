/*
 * md5_task_sgpu.cu
 *
 *  Created on: Mar 22, 2017
 *      Author: chao
 */

#include "md5_task_mgpu.h"
#include "md5_kernel.h"
#include "../md5.h"
#include "../../../common/helper_err.h"

void MD5MGPU::initImpl(config_t* args){
	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<"begin init ...\n";
		md5Config = args;
	}
	intra_Barrier();
	int buffersPerThread = md5Config->numinputs / __numLocalThreads;
	if(__localThreadId < md5Config->numinputs % __numLocalThreads){
		local_numBuffers = buffersPerThread+1;
		local_startBufferIndex = (buffersPerThread+1)*__localThreadId;
	}
	else{
		local_numBuffers = bufferPerThread;
		local_startBufferIndex = bufferPerThread*__localThreadId + md5Config->numinputs % __numLocalThreads;
	}
	local_buffer = new uint8_t[local_numBuffers*md5Config->size];
	uint8_t gbuff = md5Config->inputs;
	gbuff = gbuff+local_startBufferIndex*md5Config->size;
	for(int i=0; i<local_numBuffers;i++){
		uint8_t *p = &local_buffer[i];
		for(int j=0; j<md5Config->size; j++){
			p[j*local_numBuffers] = gbuff[i*md5Config->size +j];
		}
	}

	intra_Barrier();
	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<" finish initImpl.\n";
	}
}

void MD5MGPU::runImpl(double** runtime, int blocksize, MemType memtype){
	if(__localThreadId == 0){
		std::cout<<getCurrentTask()->getName()<<" begin run ..."<<std::endl;
	}
	Timer timer, timer0;
	double totaltime =0;

	/*
	 * create gpumem
	 */
	GpuData<uint8_t> partial_inputs(local_numBuffers*md5Config->size, memtype);
	GpuData<uint8_t> partial_out(local_numBuffers*DIGEST_SIZE, memtype);
	partial_inputs.initH(local_buffer);
	//std::cout<<md5Config->numinputs<<" "<<md5Config->size<<std::endl;
	//cudaDeviceProp prop;
	//cudaGetDeviceProperties(&prop, __deviceId);
	//std::cout<<prop.unifiedAddressing<<std::endl;
	/*
	 * copy in
	 */
	timer0.start();
	timer.start();
	partial_inputs.sync();
	double copyinTime = timer.stop();

	/*
	 * call kernel
	 */
	//if(blocksize > __blocksize)
	//	blocksize = __blocksize;
	dim3 block(blocksize, 1, 1);
	dim3 grid((local_numBuffers + block.x -1)/block.x, 1, 1);
	timer.start();
	for(int i=0; i<md5Config->iterations; i++){
		md5_process<<<grid, block, 0, __streamId>>>(
				partial_inputs.getD(),
				partial_out.getD(true),
				local_numBuffers,
				md5Config->size);
		checkCudaErr(cudaGetLastError());
		checkCudaErr(cudaStreamSynchronize(__streamId));
		checkCudaErr(cudaDeviceSynchronize());
	}
	double kernelTime = timer.stop();

	/*
	 * copy out
	 */
	timer.start();
	partial_out.sync();
	double copyoutTime = timer.stop();
	//totaltime  = timer0.stop();
	intra_Barrier();
	uint8_t *gout = md5Config->out;
	gout = gout + local_startBufferIndex*DIGEST_SIZE;
	for(int i=0; i<local_numBuffers; i++){
		uint8_t *p = partial_out.getH() + i;
		for(int j=0; j<DIGEST_SIZE; j++){
			gout[i*DIGEST_SIZE + j] = p[j*local_numBuffers];
		}
	}
	totaltime  = timer0.stop();


	//runtime[0] = kernelTime + copyinTime + copyoutTime;
	runtime[__localThreadId][0] = totaltime;
	runtime[__localThreadId][1] = kernelTime;
	runtime[__localThreadId][2] = copyinTime;
	runtime[__localThreadId][3] = copyoutTime;

	delete local_buffer;
	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<" finish runImpl.\n";
	}

}






