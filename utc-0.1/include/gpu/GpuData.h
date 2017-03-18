/*
 * GpuMemUtilities.h
 *
 *  Created on: Mar 9, 2017
 *      Author: Chao
 */

#ifndef INCLUDE_GPU_GPUDATA_H_
#define INCLUDE_GPU_GPUDATA_H_

#include "UtcGpuBasics.h"
#include "helper_cuda.h"


namespace iUtc{

enum class MemType{
	unknown= -1,
	pageable,
	pinned,
	unified
};

enum class MemStatus{
	unknown= -1,
    synced,
	host,
	device
};

template <typename T>
class GpuData{
private:
	unsigned long m_size[3];
	unsigned long m_size_inbytes;
	int m_dim;
	MemType m_memtype;

	T *m_hostPtr;
	T *m_devicePtr;

	MemStatus m_status; //

	int initmem(MemType memtype);

public:
	typedef T _Type;

	GpuData(unsigned long size, MemType memtype = MemType::pageable);
	GpuData(unsigned long size_x, unsigned long size_y, MemType memtype = MemType::pageable);
	GpuData(unsigned long size_x, unsigned long size_y, unsigned long size_z, MemType memtype = MemType::pageable);

	~GpuData();

	/*
	 * get memory address
	 */
	T *getH(){
		return m_hostPtr;
	}
	T *getD(){
		return m_devicePtr;
	}

	T *getH(bool isModified){
		if(isModified)
			m_status = MemStatus::host;
		return m_hostPtr;
	}
	T *getD(bool isModified ){
		if(isModified)
			m_status = MemStatus::device;
		return m_devicePtr;
	}

	T* getH(unsigned long offset){
		return &m_hostPtr[offset];
	}
	T* getD(unsigned long offset){
		return &m_devicePtr[offset];
	}
	T* getH(unsigned long offset, bool isModified){
		if(isModified)
			m_status = MemStatus::host;
		return &m_hostPtr[offset];
	}
	T* getD(unsigned long offset, bool isModified){
		if(isModified)
			m_status = MemStatus::device;
		return &m_hostPtr[offset];
	}

	// indicate host or device memory contents will change
	void setH(){
		m_status = MemStatus::host;
	}
	void setD(){
		m_status = MemStatus::device;
	}

	/*
	 * synchronize host/device memory
	 */
	void syncH(){
		if(m_memtype == MemType::unified)
			return;
		checkCudaRuntimeErrors(cudaMemcpy(m_devicePtr, m_hostPtr,
				m_size_inbytes, cudaMemcpyHostToDevice));
		m_status = MemStatus::synced;
	}

	void syncD(){
		if(m_memtype == MemType::unified)
			return;
		checkCudaRuntimeErrors(cudaMemcpy(m_hostPtr, m_devicePtr,
				m_size_inbytes, cudaMemcpyDeviceToHost));
		m_status = MemStatus::synced;
	}

	void sync(){
		if(m_memtype == MemType::unified)
			return;
		if(m_status == MemStatus::host)
			checkCudaRuntimeErrors(cudaMemcpy(m_devicePtr, m_hostPtr,
						m_size_inbytes, cudaMemcpyHostToDevice));
		else if(m_status == MemStatus::device)
			checkCudaRuntimeErrors(cudaMemcpy(m_devicePtr, m_hostPtr,
						m_size_inbytes, cudaMemcpyHostToDevice));
		m_status = MemStatus::synced;
	}

	void syncH(unsigned long offset, unsigned long size){
		if(m_memtype == MemType::unified)
			return;
		checkCudaRuntimeErrors(cudaMemcpy(&m_devicePtr[offset], &m_hostPtr[offset],
				size, cudaMemcpyHostToDevice));
		m_status = MemStatus::synced;
	}

	void syncD(unsigned long offset, unsigned long size){
		if(m_memtype == MemType::unified)
			return;
		checkCudaRuntimeErrors(cudaMemcpy(&m_hostPtr[offset], &m_devicePtr[offset],
				size, cudaMemcpyDeviceToHost));
		m_status = MemStatus::synced;
	}

	void sync(unsigned long offset, unsigned long size){
		if(m_memtype == MemType::unified)
			return;
		if(m_status == MemStatus::host)
			checkCudaRuntimeErrors(cudaMemcpy(&m_devicePtr[offset], &m_hostPtr[offset],
						size, cudaMemcpyHostToDevice));
		else if(m_status == MemStatus::device)
			checkCudaRuntimeErrors(cudaMemcpy(&m_devicePtr[offset], &m_hostPtr[offset],
						size, cudaMemcpyHostToDevice));
		m_status = MemStatus::synced;
	}

	/*
	 * host memory direct read/write
	 */
	T at(int index){
		return m_hostPtr[index];
	}
	void put(unsigned long index, const T value){
		m_hostPtr[index] = value;
	}

	unsigned long getSize(){
		return m_size[0]*m_size[1]*m_size[2];
	}

	unsigned long getBSize(){
		//std::cout<<m_size_inbytes<<std::endl;
		return m_size_inbytes;
	}


	/*
	 * host memory content init and fetch
	 */
	void initH(int value=0){
		memset(m_hostPtr, value, m_size_inbytes);
	}
	void initD(int value=0){
		checkCudaRuntimeErrors(cudaMemset(m_devicePtr, value, m_size_inbytes));
	}

	void initH(const T* other){
		memcpy(m_hostPtr, other, m_size_inbytes);
	}
	void initD(const T* other){
		checkCudaRuntimeErrors(cudaMemcpy(m_devicePtr, other,
				m_size_inbytes, cudaMemcpyDefault));
	}

	void fetchH(T* dest){
		memcpy(dest, m_hostPtr, m_size_inbytes);
	}
	void fetchD(T* dest){
		checkCudaRuntimeErrors(cudaMemcpy(dest, m_devicePtr,
				m_size_inbytes, cudaMemcpyDefault));
	}




};

}

#include "GpuData.inc"


#endif /* INCLUDE_GPU_GPUDATA_H_ */
