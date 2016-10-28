/*
 * UtcGpuContext.cc
 *
 *  Created on: Oct 21, 2016
 *      Author: chao
 */

#include "gpu/UtcGpuBasics.h"
#include "gpu/UtcGpuContext.h"

namespace iUtc{

UtcGpuContext::UtcGpuContext(int gpuId, cudaCtxMapMode ctxMode){
	m_cudaCtxMapMode = ctxMode;
	m_utcGpuId = gpuId;
	m_cudaDeviceId = m_cudaDevice = CudaDeviceManager::getCudaDeviceManager().getCudaDeviceId(gpuId);
}

UtcGpuContext::~UtcGpuContext(){
	m_cudaCtxMapMode = 0;
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
		cudaSetDevice(m_cudaDeviceId);

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
		cuCtxGetCurrent(&m_cudaContextBound);

		/*
		 * get the cuda stream handle for this host thread
		 */
		if(ENABLE_CONCURRENT_CUDA_KERNEL){
			// cudaStreamDefault, cudaStreamNonBlocking
			cudaStreamCreateWithFlags(&m_cudaStreamBound, cudaStreamDefault);

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
		else
			m_cudaStreamBound = 0; // stream 0 or default stream
		break;
	default:
		break;
	}

	return;
}// end ctxInit

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
			//cudaStreamSynchronize(m_cudaStreamBound);
			cudaStreamDestroy(m_cudaStreamBound);
		}
	}
}// end ctxDestroy

}// end namespace iUtc



