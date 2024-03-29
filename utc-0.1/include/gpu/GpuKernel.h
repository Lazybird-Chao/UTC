/*
 * GpuKernel.h
 *
 *  Created on: Nov 1, 2016
 *      Author: chao
 */

#ifndef UTC_GPUKERNEL_H_
#define UTC_GPUKERNEL_H_

#include "cuda_runtime.h"

#include "UtcGpuContext.h"

namespace iUtc{

enum class ArgType{
	value = 0,
	in =1,
	out =2,
	io = 3
};

class GpuKernel{
public:
	GpuKernel();

	~GpuKernel();

	int setNumArgs(int numArgs);
	int getNumArgs();

	template<typename T>
	int setArgs(int idx, T &arg, ArgType argtype=ArgType::value, size_t size=0);

	int setGridDim(int d1);
	int setGridDim(int d1, int d2);
	int setGridDim(int d1, int d2, int d3);

	int setBlockDim(int d1);
	int setBlockDim(int d1, int d2);
	int setBlockDim(int d1, int d2, int d3);

	int setSharedMemSize(size_t size);

	int launchKernel(const void* kernel_fun, bool async = false);
	int launchKernel(const void* kernel_fun, cudaStream_t stream, bool async = false);

	int syncKernel();
	int syncKernel(cudaStream_t stream);

	UtcGpuContext *getUtcGpuContext();

private:
	bool m_useCudaUnifiedMemForArgTrans;

	dim3 m_cudaGridDim;
	dim3 m_cudaBlockDim;
	size_t m_sharedMemSize;

	int m_numArgs;
	void **m_args;
	int *m_argSizes;
	int *m_argTypes;
	size_t *m_argReferSizes;

	UtcGpuContext *m_crtUtcGpuCtx;
	cudaStream_t m_cudaStream;

};


template<typename T>
int GpuKernel::setArgs(int idx, T &arg, ArgType argtype, size_t size){
	m_args[idx] = (void*)&arg;

	switch(argtype){
	case ArgType::value:
		m_argTypes[idx] = 0;
		m_argSizes[idx] = sizeof(T);
		m_argReferSizes[idx] = 0;
		break;
	case ArgType::in:
		m_argTypes[idx] = 1;
		m_argSizes[idx] = sizeof(T);
		m_argReferSizes[idx] = size;
		break;
	case ArgType::out:
		m_argTypes[idx] = 2;
		m_argSizes[idx] = sizeof(T);
		m_argReferSizes[idx] = size;
		break;
	case ArgType::io:
		m_argTypes[idx] = 3;
		m_argSizes[idx] = sizeof(T);
		m_argReferSizes[idx] = size;
		break;
	}

	return 0;
}

}



#endif /* UTC_GPUKERNEL_H_ */
