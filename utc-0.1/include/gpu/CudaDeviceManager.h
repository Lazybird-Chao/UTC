/*
 * CudaDeviceManager.h
 *
 *  Created on: Oct 24, 2016
 *      Author: chao
 */

#ifndef UTC_GPU_CUDADEVICEMANAGER_H_
#define UTC_GPU_CUDADEVICEMANAGER_H_

#include "UtcGpuBasics.h"

#include "cuda_runtime.h"

namespace iUtc{

typedef struct cudaDeviceInfo{
	int				devID;
	cudaDeviceProp 	*deviceProp;
	int				smMajor;
	int				smMinor;
	bool			inited;
	bool			reseted;

} cudaDeviceInfo;

class CudaDeviceManager{
public:
	~CudaDeviceManager();

	static CudaDeviceManager& getCudaDeviceManager();

	int getNumDevices();

	cudaDeviceInfo getDeviceInfo(int devId);

	int getCudaDeviceId(int devId);

	int getAvailableGpu(int threadLocalRank);

	/*
	 * these init and reset should only be called once in a process;
	 * so it only applied to "cudaCtxMapToDevice" mode
	 * it will init the cuda primary context on each gpu;
	 * as well as reset this primary context on each gpu;
	 */
	void initDevice(int devId);
	void resetDevice(int devId);

	static int				runtimeMajor;
	static int				runtimeMinor;
	static int				driverMajor;
	static int				driverMinor;


private:
	CudaDeviceManager();

	static CudaDeviceManager* m_managerInstance;

	int m_numDevicesForUse;
	int m_numTotalDevices;
	cudaDeviceInfo m_deviceInfoArray[MAX_DEVICE_PER_NODE];

	int m_deviceForUseArray[MAX_DEVICE_PER_NODE];

};

}// end namespace iUtc



#endif /* UTC_GPU_CUDADEVICEMANAGER_H_ */
