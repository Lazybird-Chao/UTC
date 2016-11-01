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

}
