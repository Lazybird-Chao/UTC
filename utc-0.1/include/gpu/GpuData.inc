/*
 * GpuData.inc
 *
 *  Created on: Mar 9, 2017
 *      Author: Chao
 */

#ifndef INCLUDE_GPU_GPUDATA_INC_
#define INCLUDE_GPU_GPUDATA_INC_

#include "cuda_runtime.h"
#include <iostream>


namespace iUtc{

template<typename T>
GpuData<T>::GpuData(MemType memtype){
	m_dim = 0;
	m_size[0] = 0;
	m_size[1] = 0;
	m_size[2] = 0;
	m_size_inbytes = 0;
	isinitialized = false;
	iscleaned = false;
	m_hostPtr = m_devicePtr = NULL;
	m_memtype = memtype;
	m_status = MemStatus::unknown;
}

template <typename T>
GpuData<T>::GpuData(unsigned long size, MemType memtype ){
	m_dim = 1;
	m_size[0] = size;
	m_size[1] = 1;
	m_size[2] = 1;
	m_size_inbytes = m_size[0]*m_size[1]*m_size[2]*sizeof(T);
	isinitialized = false;
	iscleaned = false;
	initmem(memtype);
}

template <typename T>
GpuData<T>::GpuData(unsigned long size_x, unsigned long size_y, MemType memtype){
	m_dim = 2;
	m_size[0] = size_x;
	m_size[1] = size_y;
	m_size[2] = 1;
	m_size_inbytes = m_size[0]*m_size[1]*m_size[2]*sizeof(T);
	isinitialized = false;
	iscleaned = false;
	initmem(memtype);
}

template <typename T>
GpuData<T>::GpuData(unsigned long size_x, unsigned long size_y, unsigned long size_z, MemType memtype){
	m_dim = 3;
	m_size[0] = size_x;
	m_size[1] = size_y;
	m_size[2] = size_z;
	m_size_inbytes = m_size[0]*m_size[1]*m_size[2]*sizeof(T);
	isinitialized = false;
	iscleaned = false;
	initmem(memtype);

}

template<typename T>
int GpuData<T>::initmem(MemType memtype){
	m_memtype = memtype;
	switch (m_memtype){
	case MemType::pageable:
		m_hostPtr = new T[m_size[0]*m_size[1]*m_size[2]];
		checkCudaRuntimeErrors(cudaMalloc(&m_devicePtr, m_size_inbytes));
		m_status = MemStatus::host;
		break;
	case MemType::pinned:
		checkCudaRuntimeErrors(cudaMallocHost(&m_hostPtr, m_size_inbytes));
		checkCudaRuntimeErrors(cudaMalloc(&m_devicePtr, m_size_inbytes));
		m_status = MemStatus::host;
		break;
	case MemType::unified:
		if(getCurrentUtcGpuCtx()->getCurrentDeviceAttr(cudaDevAttrManagedMemory) ==1){
			//std::cout<<"using cuda managed for umem"<<std::endl;
			checkCudaRuntimeErrors(cudaMallocManaged(&m_devicePtr, m_size_inbytes));
		}
		else{
			//std::cout<<"using pinned zero-copy for umem"<<std::endl;
			checkCudaRuntimeErrors(cudaMallocHost(&m_devicePtr, m_size_inbytes));
		}
		m_hostPtr = m_devicePtr;
		m_status = MemStatus::synced;
		break;
	default:
		std::cerr<<"ERROR, undefined gpu memory type !!!"<<std::endl;
		break;
	}
	isinitialized = true;
	iscleaned = false;
	return 0;
}

template<typename T>
int GpuData<T>::initMem(unsigned long size_x, MemType memtype){
	m_dim = 1;
	return initMem(size_x, 1, 1, memtype);
}

template<typename T>
int GpuData<T>::initMem(unsigned long size_x, unsigned long size_y, MemType memtype){
	m_dim = 2;
	return initMem(size_x, size_y, 1, memtype);
}

template<typename T>
int GpuData<T>::initMem(unsigned long size_x, unsigned long size_y, unsigned long size_z, MemType memtype){
	if(m_dim==0)
		m_dim = 3;
	if(!isinitialized){
		m_size[0] = size_x;
		m_size[1] = size_y;
		m_size[2] = size_z;
		m_size_inbytes = m_size[0]*m_size[1]*m_size[2]*sizeof(T);
		initmem(memtype);
	}
	return 0;
}

template <typename T>
GpuData<T>::~GpuData(){
	if(!iscleaned)
		cleanMem();
	m_hostPtr = m_devicePtr = NULL;
}

template <typename T>
void GpuData<T>::cleanMem(){
	switch (m_memtype){
	case MemType::pageable:
		delete[] m_hostPtr;
		checkCudaRuntimeErrors(cudaFree(m_devicePtr));
		break;
	case MemType::pinned:
		checkCudaRuntimeErrors(cudaFreeHost(m_hostPtr));
		checkCudaRuntimeErrors(cudaFree(m_devicePtr));
		break;
	case MemType::unified:
		if(getCurrentUtcGpuCtx()->getCurrentDeviceAttr(cudaDevAttrManagedMemory) ==1){
			checkCudaRuntimeErrors(cudaFree(m_devicePtr));
		}
		else
			checkCudaRuntimeErrors(cudaFreeHost(m_hostPtr));
		break;
	}
	iscleaned = true;
}

}
#endif /* INCLUDE_GPU_GPUDATA_INC_ */
