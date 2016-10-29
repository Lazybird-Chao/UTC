/*
 * CudaDeviceManager.cc
 *
 *  Created on: Oct 27, 2016
 *      Author: chao
 */

#include "gpu/CudaDeviceManager.h"
#include "gpu/UtcGpuBasics.h"

#include "cuda_runtime.h"
#include "cuda.h"

namespace iUtc{

CudaDeviceManager* m_managerInstance = nullptr;
CudaDeviceManager::CudaDeviceManager(){
	int driverVersion, runtimeVersion;
	cudaDriverGetVersion(&driverVersion);
	cudaRuntimeGetVersion(&runtimeVersion);
	driverMajor = driverVersion/1000;
	driverMinor = (driverVersion%100)/10;
	runtimeMajor = runtimeVersion/1000;
	runtimeMinor = (runtimeVersion%100)/10;

	cudaGetDeviceCount(&m_numTotalDevices);
	m_numDevicesForUse = 0;
	for(int i=0; i<MAX_DEVICE_PER_NODE;i++)
		m_deviceForUseArray[i] = -1;
	for(int i=0; i<m_numTotalDevices; i++){
		cudaSetDevice(i);
		cudaDeviceProp *devProp = (cudaDeviceProp*) malloc(sizeof(cudaDeviceProp));
		cudaGetDeviceProperties(devProp, i);
		/*
		 * we only record GPU capability >=2
		 */
		if(devProp->major<2){
			free(devProp);
			cudaDeviceReset();
			continue;
		}
		else{
			m_deviceForUseArray[m_numDevicesForUse] = i;
			m_deviceInfoArray[m_numDevicesForUse].devID = i;
			m_deviceInfoArray[m_numDevicesForUse].smMajor = devProp->major;
			m_deviceInfoArray[m_numDevicesForUse].smMinor = devProp->minor;
			m_deviceInfoArray[m_numDevicesForUse].deviceProp = devProp;
			m_deviceInfoArray[m_numDevicesForUse].inited = false;
			m_deviceInfoArray[m_numDevicesForUse].reseted = false;
			m_numDevicesForUse++;
			cudaDeviceReset();
		}
	}

}

CudaDeviceManager::~CudaDeviceManager(){
	m_numTotalDevices = 0;
	for(int i=0; i<MAX_DEVICE_PER_NODE;i++)
		m_deviceForUseArray[i] = -1;
	for(int i=0; i<m_numDevicesForUse;i++)
		free(m_deviceInfoArray[i].deviceProp);
	m_managerInstance = nullptr;
}

CudaDeviceManager& CudaDeviceManager::getCudaDeviceManager(){
	if(m_managerInstance == nullptr)
		m_managerInstance = new CudaDeviceManager();
	return *m_managerInstance;
}

int CudaDeviceManager::getNumDevices(){
	return m_numDevicesForUse;
}

cudaDeviceInfo CudaDeviceManager::getDeviceInfo(int devId){
	return m_deviceInfoArray[devId];
}

void CudaDeviceManager::initDevice(int devId){
	if(m_deviceInfoArray[devId].inited == false &&
			m_deviceInfoArray[devId].reseted == false){
		/*
		 * this devId is not the gpu device id managed by cuda.
		 * the real cuda id is m_deviceForUseArray[devId]
		 */
		cudaSetDevice(m_deviceForUseArray[devId]);
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

		m_deviceInfoArray[devId].inited= true;
	}

}

void CudaDeviceManager::resetDevice(int devId){
	if(m_deviceInfoArray[devId].inited==true &&
			m_deviceInfoArray[devId].reseted==false){
		cudaSetDevice(m_deviceForUseArray[devId]);
		cudaDeviceReset();
		m_deviceInfoArray[devId].reseted = true;
	}
}

inline int CudaDeviceManager::getCudaDeviceId(int devId){
	return m_deviceForUseArray[devId];
}

inline int CudaDeviceManager::getAvailableGpu(int threadLocalRank){
	return threadLocalRank % m_numDevicesForUse;
}

}// end namespace iUtc


