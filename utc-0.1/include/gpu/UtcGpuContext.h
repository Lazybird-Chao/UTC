/*
 * UtcGpuContext.h
 *
 *  Created on: Oct 21, 2016
 *      Author: chao
 */

#ifndef UTC_GPU_UTCGPUCONTEXT_H_
#define UTC_GPU_UTCGPUCONTEXT_H_

#include "cuda_runtime.h"
#include "cuda.h"

namespace iUtc{

enum class cudaCtxMapMode{
	unknown =0,
	cudaCtxMapToThread,
	cudaCtxMapToTask,
	cudaCtxMapToDevice
};


/*
 * this context is used by each task thread.
 * Each thread of gpu task will have such a context obj.
 * It's not the cuda context.
 */
class UtcGpuContext{
public:
	UtcGpuContext(int gpuId, cudaCtxMapMode ctxMode = cudaCtxMapMode::cudaCtxMapToDevice);

	~UtcGpuContext();

	void ctxInit();

	void ctxDestroy();

	CUcontext* getCudaContext();

	int getUtcGpuId();

	int getCudaDeviceId();

	cudaStream_t getBoundStream();

	cudaStream_t getNewStream();
	void destroyStream(cudaStream_t &stream);

	cudaEvent_t getNewEvent();
	void destroyEvent(cudaEvent_t &event);

	int getCurrentDeviceAttr(cudaDeviceAttr attr);
	int getCurrentDeviceAttr(cudaDeviceAttr attr, int cudaDevId);

private:
	// the actual related cuda context
	CUcontext	m_cudaContextBound;

	// the cuda stream handle
	cudaStream_t m_cudaStreamBound;

	// the bind gpu device id
	int		m_cudaDeviceId;    //the real gpuid that can be use by cudaSetDevice()
	CUdevice	m_cudaDevice;

	int 	m_utcGpuId;  // different from the m_cudaDeviceId

	//
	cudaCtxMapMode	m_cudaCtxMapMode;

};

}



#endif /* INCLUDE_GPU_UTCGPUCONTEXT_H_ */
