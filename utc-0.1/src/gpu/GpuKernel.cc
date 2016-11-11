/*
 * GpuKernel.cc
 *
 *  Created on: Nov 1, 2016
 *      Author: chao
 */

#include "GpuKernel.h"
#include "GpuTaskUtilities.h"
#include "helper_cuda.h"

namespace iUtc{

GpuKernel::GpuKernel(){
	m_cudaGridDim = dim3(1,1,1);
	m_cudaBlockDim = dim3(1,1,1);

	m_sharedMemSize = 0;
	m_numArgs = 0;
	m_args = nullptr;
	m_argSizes = nullptr;
	m_argTypes = nullptr;
	m_argReferSizes = nullptr;

	m_crtUtcGpuCtx = getCurrentUtcGpuCtx();
	m_cudaStream = m_crtUtcGpuCtx->getBoundStream();

	m_useCudaUnifiedMemForArgTrans =false;
}

GpuKernel::~GpuKernel(){
	m_cudaGridDim = dim3(1,1,1);
	m_cudaBlockDim = dim3(1,1,1);

	m_sharedMemSize = 0;
	m_numArgs = 0;
	if(m_args != nullptr)
		free(m_args);
	m_args = nullptr;
	if(m_argSizes != nullptr)
		free(m_argSizes);
	m_argSizes = nullptr;
	if(m_argTypes != nullptr)
		free(m_argTypes);
	m_argTypes = nullptr;
	if(m_argReferSizes != nullptr)
			free(m_argReferSizes);
	m_argReferSizes = nullptr;
}

int GpuKernel::setNumArgs(int numArgs){
	m_numArgs = numArgs;
	if(m_args != nullptr)
		free(m_args);
	m_args = (void**)malloc(sizeof(void*)*numArgs);
	if(m_argSizes != nullptr)
			free(m_argSizes);
	m_argSizes = (int*)malloc(sizeof(int)*numArgs);
	if(m_argTypes != nullptr)
			free(m_argTypes);
	m_argTypes = (int*)malloc(sizeof(int)*numArgs);
	if(m_argReferSizes != nullptr)
				free(m_argReferSizes);
	m_argReferSizes = (size_t*)malloc(sizeof(size_t)*numArgs);

	return 0;
}

int GpuKernel::getNumArgs(){
	return m_numArgs;
}


int GpuKernel::setSharedMemSize(size_t size){
	m_sharedMemSize = size;
	return 0;
}

int GpuKernel::setGridDim(int d1){
	m_cudaGridDim = dim3(d1, 1,1);
	return 0;
}

int GpuKernel::setGridDim(int d1, int d2){
	m_cudaGridDim = dim3(d1, d2,1);
	return 0;
}

int GpuKernel::setGridDim(int d1, int d2, int d3){
	m_cudaGridDim = dim3(d1, d2, d3);
	return 0;
}

int GpuKernel::setBlockDim(int d1){
	m_cudaBlockDim = dim3(d1, 1,1);
	return 0;
}

int GpuKernel::setBlockDim(int d1, int d2){
	m_cudaBlockDim = dim3(d1, d2,1);
	return 0;
}

int GpuKernel::setBlockDim(int d1, int d2, int d3){
	m_cudaBlockDim = dim3(d1, d2,d3);
	return 0;
}

int GpuKernel::launchKernel(const void* kernel_fun, bool async){
	checkCudaRuntimeErrors(cudaConfigureCall(m_cudaGridDim, m_cudaBlockDim, m_sharedMemSize, m_cudaStream));
	size_t offset = 0;
	for(int i=0; i< m_numArgs; i++){
		checkCudaRuntimeErrors(cudaSetupArgument(m_args[i], m_argSizes[i], offset));
		offset += m_argSizes[i];
	}

	checkCudaRuntimeErrors(cudaLaunch(kernel_fun));

	if(!async)
		checkCudaRuntimeErrors(cudaStreamSynchronize(m_cudaStream));

	return 0;
}

int GpuKernel::syncKernel(){
	checkCudaRuntimeErrors(cudaStreamSynchronize(m_cudaStream));
	return 0;
}


}// end namespace iUtc


