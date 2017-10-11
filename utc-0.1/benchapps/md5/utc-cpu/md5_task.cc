/*
 * md5_task.cc
 *
 *  Created on: Oct 10, 2017
 *      Author: chaoliu
 */
#include "md5.h"
#include "md5_compute.h"
#include "task.h"

thread_local long MD5Worker::local_numBuffers;
thread_local long MD5Worker::local_startBufferIndex;

void MD5Worker::initImpl(config_t *args){
	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<"begin init ...\n";
		md5Config = args;
	}
	__fastIntraSync.wait();
	long buffersPerThread = md5Config->numinputs / __numLocalThreads;
	if(__localThreadId < md5Config->numinputs % __numLocalThreads){
		local_numBuffers = buffersPerThread+1;
		local_startBufferIndex = (buffersPerThread+1)*__localThreadId;
	}
	else{
		local_numBuffers = buffersPerThread;
		local_startBufferIndex = buffersPerThread*__localThreadId + md5Config->numinputs % __numLocalThreads;
	}
	__fastIntraSync.wait();
	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<" finish initImpl.\n";
	}
}

void MD5Worker::runImpl(double runtime[][1]){
	if(__localThreadId == 0){
		std::cout<<getCurrentTask()->getName()<<" begin run ..."<<std::endl;
	}
	Timer timer;

	timer.start();
	for(int i = 0; i < md5Config->iterations; i++){
		uint8_t *in = &(md5Config->inputs[local_startBufferIndex*md5Config->size]);
		uint8_t *out = &(md5Config->out[local_startBufferIndex*DIGEST_SIZE]);
		long j = 0;
		while(j < local_numBuffers){
			//if(j%1000 == 0)
			//	std::cout<<j/1000<<std::endl;
			process(&in[j*md5Config->size], out+j*DIGEST_SIZE, md5Config->size);
			j++;
		}
	}
	double totaltime = timer.stop();
	runtime[__localThreadId][0] = totaltime;
	__fastIntraSync.wait();

	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<" finish runImpl.\n";
	}
}

void MD5Worker::process(uint8_t *in, uint8_t *out, long bufsize){
	MD5_CTX context;
	uint8_t digest[16];

	MD5_Init(&context);
	MD5_Update(&context, in, bufsize);
	//MD5_CTX *ctx = &context;
	//std::cout<<ctx->a<<" "<<ctx->b<<" "<<ctx->c<<" "<<ctx->d<<std::endl;
	MD5_Final((unsigned char*)digest, &context);

	memcpy(out, digest, DIGEST_SIZE);
}

