#include "gpu/GpuTaskUtilities.h"
#include "gpu/CudaDeviceManager.h"
#include "TaskManager.h"

namespace iUtc{

UtcGpuContext* getCurrentUtcGpuCtx(){
		static thread_local UtcGpuContext *crtUtcGpuCtx=nullptr;
		if(crtUtcGpuCtx == nullptr){
			crtUtcGpuCtx = TaskManager::getTaskInfo()->gpuSpecInfo.utcGpuCtx;

		}
		return crtUtcGpuCtx;
	}

int getCurrentUtcGpuId(){
	static thread_local UtcGpuContext *crtUtcGpuCtx=nullptr;
	if(crtUtcGpuCtx == nullptr){
		crtUtcGpuCtx = TaskManager::getTaskInfo()->gpuSpecInfo.utcGpuCtx;

	}
	return crtUtcGpuCtx->getUtcGpuId();
}

int getCurrentCudaDeviceId(){
	static thread_local UtcGpuContext *crtUtcGpuCtx=nullptr;
		if(crtUtcGpuCtx == nullptr){
			crtUtcGpuCtx = TaskManager::getTaskInfo()->gpuSpecInfo.utcGpuCtx;
		}
		return crtUtcGpuCtx->getCudaDeviceId();
}

cudaStream_t getCurrentStream(){
	static thread_local cudaStream_t crtStream = NULL;
	if(crtStream == NULL){
		(TaskManager::getTaskInfo()->gpuSpecInfo.utcGpuCtx)->getDefaultStream();
	}
	return crtStream;
}

}
