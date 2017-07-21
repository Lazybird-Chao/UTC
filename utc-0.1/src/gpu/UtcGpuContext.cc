/*
 * UtcGpuContext.cc
 *
 *  Created on: Oct 21, 2016
 *      Author: chao
 */

#include "UtcGpuBasics.h"
#include "UtcGpuContext.h"
#include "CudaDeviceManager.h"
#include "helper_cuda.h"
#include <iostream>

namespace iUtc{

UtcGpuContext::UtcGpuContext(int gpuId, cudaCtxMapMode ctxMode){
	m_cudaCtxMapMode = ctxMode;
	m_utcGpuId = gpuId;
	m_cudaDeviceId = m_cudaDevice = CudaDeviceManager::getCudaDeviceManager().getCudaDeviceId(gpuId);
}

UtcGpuContext::~UtcGpuContext(){
	m_cudaCtxMapMode = cudaCtxMapMode::unknown;
	m_utcGpuId = -1;
	m_cudaDeviceId = m_cudaDevice = -1;
}

void UtcGpuContext::ctxInit(){
	switch (m_cudaCtxMapMode){
	case cudaCtxMapMode::cudaCtxMapToThread:
		break;
	case cudaCtxMapMode::cudaCtxMapToTask:
		break;
	case cudaCtxMapMode::cudaCtxMapToDevice:
		/*
		 * bind cuda device primary ctx to this host thread
		 */
		checkCudaRuntimeErrors(cudaSetDevice(m_cudaDeviceId));
		std::cout<<"cuda device id:"<<m_cudaDeviceId<<std::endl;
		/*
		 * set flag for device, should be done in other places,
		 * here we just bind the primary cuda ctx to this host thread
		 * possible flags: cudaDeviceScheduleAtuo
		 * 				   cudaDeviceScheduleSpin
		 * 				   cudaDeviceScheduleYield
		 * 				   cudaDeviceScheduleBlockingSync
		 * 				   cudaDeviceMapHost
		 * 				   cudaDeviceLmemResizeToMax
		 */
		//cudaSetDeviceFlags();

		/*
		 * get the current primary cuda ctx
		 */
		checkCudaDriverErrors(cuCtxGetCurrent(&m_cudaContextBound));

		/*
		 * get the cuda stream handle for this host thread
		 */
		if(ENABLE_CONCURRENT_CUDA_KERNEL){
			// cudaStreamDefault, cudaStreamNonBlocking
			if(USING_NONBLOCKING_STREAM){
				checkCudaRuntimeErrors(
					cudaStreamCreateWithFlags(&m_cudaStreamBound, cudaStreamNonBlocking));
			}
			else{
				checkCudaRuntimeErrors(cudaStreamCreateWithFlags(&m_cudaStreamBound, cudaStreamDefault));
				//m_cudaStreamBound = cudaStreamPerThread;
			}
			/* create stream with flag and priority
			 */
			 /*int hp, lp;
			 cudaDeviceGetStreamPriorityRange(&lp, &hp);
			 int priority= 1;
			 if(priority <hp)
			 	 priority = hp;
		     if(priority > lp)
		     	 priority = lp;
			 cudaStreamCreateWithPriorit(&m_cudaStreamBound, cudaStreamDefault,
											priority);
											*/
		}
		else{
			//m_cudaStreamBound = cudaStreamLegacy; // stream 0 or default legacy stream
			m_cudaStreamBound = NULL;
		}
		break;
	default:
		break;
	}

	return;
}// end ctxInit

int UtcGpuContext::setNonblockingDefaultStream(){
	/*
	 * when update the default stream for a task thread, you may also
	 * need update "__streamId" pre-built var
	 */
	checkCudaRuntimeErrors(
			cudaStreamCreateWithFlags(&m_cudaStreamBound, cudaStreamNonBlocking));
	return 0;
}

void UtcGpuContext::ctxDestroy(){
	switch(m_cudaCtxMapMode){
	case cudaCtxMapMode::cudaCtxMapToThread:
		break;
	case cudaCtxMapMode::cudaCtxMapToTask:
		break;
	case cudaCtxMapMode::cudaCtxMapToDevice:
		/*
		 *  the primary context should not be destroyed here
		 */

		/*
		 *  if not the default stream, destroy it
		 */
		if(ENABLE_CONCURRENT_CUDA_KERNEL){
			/*
			 * may need to call stream sync before destroy it
			 */
			cudaStreamSynchronize(m_cudaStreamBound);
			if(m_cudaStreamBound != NULL )
				checkCudaRuntimeErrors(cudaStreamDestroy(m_cudaStreamBound));
		}
	}
}// end ctxDestroy


CUcontext* UtcGpuContext::getCudaContext(){
	return &m_cudaContextBound;
}

int UtcGpuContext::getCudaDeviceId(){
	return m_cudaDeviceId;
}

int UtcGpuContext::getUtcGpuId(){
	return m_utcGpuId;
}

cudaStream_t UtcGpuContext::getBoundStream(){
	return m_cudaStreamBound;
}

cudaStream_t UtcGpuContext::getDefaultStream(){
	return m_cudaStreamBound;
}

cudaStream_t UtcGpuContext::getNewStream(){
	cudaStream_t stream;
	checkCudaRuntimeErrors(cudaStreamCreateWithFlags(&stream, cudaStreamDefault));
	return stream;
}

void UtcGpuContext::destroyStream(cudaStream_t &stream){
	checkCudaRuntimeErrors(cudaStreamDestroy(stream));
}

cudaEvent_t UtcGpuContext::getNewEvent(){
	cudaEvent_t event;
	checkCudaRuntimeErrors(cudaEventCreate(&event));
	return event;
}

void UtcGpuContext::destroyEvent(cudaEvent_t &event){
	checkCudaRuntimeErrors(cudaEventDestroy(event));
}

int UtcGpuContext::getCurrentDeviceAttr(cudaDeviceAttr attr){
	return getCurrentDeviceAttr(attr, m_cudaDeviceId);
}

int UtcGpuContext::getCurrentDeviceAttr(cudaDeviceAttr attr, int cudaDevId){
	int value;
	checkCudaRuntimeErrors(cudaDeviceGetAttribute(&value, attr, cudaDevId));
	return value;
}

}// end namespace iUtc



