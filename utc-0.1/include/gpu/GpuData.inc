/*
 * GpuData.inc
 *
 *  Created on: Mar 9, 2017
 *      Author: Chao
 */

#ifndef INCLUDE_GPU_GPUDATA_INC_
#define INCLUDE_GPU_GPUDATA_INC_

#include "cuda_runtime.h"
#include "GpuTaskUtilites.h"
#include <iostream>


namespace iUtc{

template <typename T>
GpuData<T>::GpuData(size_t size, MemType memtype = MemType::pagable){
	m_dim = 1;
	GpuData(size, 1, 1, memtype);
}

template <typename T>
GpuData<T>::GpuData(size_t size_x, size_t size_y, MemType memtype = MemType::pagable){
	m_dim =2;
	GpuData(size_x, size_y, 1, memtype);
}

template <typename T>
GpuData<T>::GpuData(size_t size_x, size_t size_y, size_t size_z, MemType memtype = MemType::pagable){
	m_size[0] = size_x;
	m_size[1] = size_y;
	m_size[2] = size_z;

	m_size_inbytes = m_size[0]*m_size[1]*m_size[2]*sizeof(T);
	m_dim = 3;
	m_memtype = memtype;

	switch (m_memtype){
	case MemType::pageable:
		m_hostPtr = new T[m_size[0]*m_size[1]*m_size[2]];
		checkCudaDriverErrors(cudaMalloc(&m_devicePtr, m_size_inbytes));
		m_status = MemStatus::host;
		break;
	case MemType::pinned:
		checkCudaDriverErrors(cudaMallocHost(&m_hostPtr, m_size_inbytes));
		checkCudaDriverErrors(cudaMalloc(&m_devicePtr, m_size_inbytes));
		m_status = MemStatus::host;
		break;
	case MemType::unified:
		if(getCurrentUtcGpuCtx()->getCurrentDeviceAttr(cudaDevAttrManagedMemory) ==1){
			//std::cout<<"using cuda managed for umem"<<std::endl;
			checkCudaErr(cudaMallocManaged(&m_devicePtr, m_size_inbytes));
		}
		else{
			//std::cout<<"using pinned zero-copy for umem"<<std::endl;
			checkCudaErr(cudaMallocHost(&m_devicePtr, m_size_inbytes));
		}
		m_hostPtr = m_devicePtr;
		m_status = MemStatus::synced;
		break;
	default:
		std::cerr<<"ERROR, undefined gpu memory type !!!"<<std::endl;
		break;
	}

}

template <typename T>
GpuData<T>::~GpuData(){

	switch (m_memtype){
	case MemType::pageable:
		delete[] m_hostPtr;
		checkCudaDriverErrors(cudaFree(m_devicePtr));
		break;
	case MemType::pinned:
		checkCudaDriverErrors(cudaFree(m_hostPtr));
		checkCudaDriverErrors(cudaFree(m_devicePtr));
		break;
	case MemType::unified:
		checkCudaDriverErrors(cudaFree(m_devicePtr));
		break;
	}
	m_hostPtr = m_devicePtr = NULL;
}


}
#endif /* INCLUDE_GPU_GPUDATA_INC_ */
