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
	size_t m_size[3];
	size_t m_size_inbytes;
	int dim;
	MemType m_memtype;

	T *m_hostPtr;
	T *m_devicePtr;

	MemStatus m_status; //

public:
	typedef T _Type;

	GpuData(size_t size, MemType memtype = MemType::pagable);
	GpuData(size_t size_x, size_t size_y, MemType memtype = MemType::pagable);
	GpuData(size_t size_x, size_t size_y, size_t size_z, MemType memtype = MemType::pagable);

	~GpuData();

	/*
	 * get memory address
	 */
	T *getH(){
		return m_hostPtr;
	}
	T *getH(bool isModified){
		if(isModified)
			m_status = MemStatus::host;
		return m_hostPtr;
	}
	T *getD(bool bool isModified ){
		if(isModified)
			m_status = MemStatus::device;
		return m_devicePtr;
	}

	T* getH(size_t offset){
		return &m_hostPtr[offset];
	}
	T* getD(size_t offset){
		return &m_devicePtr[offset];
	}
	T* getH(size_t offset, bool isModified){
		if(isModified)
			m_status = MemStatus::host;
		return &m_hostPtr[offset];
	}
	T* getD(size_t offset, bool isModified){
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
		checkCudaDriverErrors(cudaMemcpy(m_devicePtr, m_hostPtr,
				m_size_inbytes, cudaMemcpyHostToDevice));
		m_status = MemStatus::synced;
	}

	void syncD(){
		if(m_memtype == MemType::unified)
			return;
		checkCudaDriverErrors(cudaMemcpy(m_hostPtr, m_devicePtr,
				m_size_inbytes, cudaMemcpyDeviceToHost));
		m_status = MemStatus::synced;
	}

	void sync(){
		if(m_memtype == MemType::unified)
			return;
		if(m_status == MemStatus::host)
			checkCudaDriverErrors(cudaMemcpy(m_devicePtr, m_hostPtr,
						m_size_inbytes, cudaMemcpyHostToDevice));
		else if(m_status == MemStatus::device)
			checkCudaDriverErrors(cudaMemcpy(m_devicePtr, m_hostPtr,
						m_size_inbytes, cudaMemcpyHostToDevice));
		m_status = MemStatus::synced;
	}

	void syncH(size_t offset, size_t size){
		if(m_memtype == MemType::unified)
			return;
		checkCudaDriverErrors(cudaMemcpy(&m_devicePtr[offset], &m_hostPtr[offset],
				size, cudaMemcpyHostToDevice));
		m_status = MemStatus::synced;
	}

	void syncD(size_t offset, size_t size){
		if(m_memtype == MemType::unified)
			return;
		checkCudaDriverErrors(cudaMemcpy(&m_hostPtr[offset], &m_devicePtr[offset],
				size, cudaMemcpyDeviceToHost));
		m_status = MemStatus::synced;
	}

	void sync(size_t offset, size_t size){
		if(m_memtype == MemType::unified)
			return;
		if(m_status == MemStatus::host)
			checkCudaDriverErrors(cudaMemcpy(&m_devicePtr[offset], &m_hostPtr[offset],
						size, cudaMemcpyHostToDevice));
		else if(m_status == MemStatus::device)
			checkCudaDriverErrors(cudaMemcpy(&m_devicePtr[offset], &m_hostPtr[offset],
						size, cudaMemcpyHostToDevice));
		m_status = MemStatus::synced;
	}

	/*
	 * host memory direct read/write
	 */
	T at(int index){
		return m_hostPtr[index];
	}
	void put(size_t index, const T value){
		m_hostPtr[index] = value;
	}

	size_t getSize(){
		return m_size[0]*m_size[1]*m_size[2];
	}

	size_t getBSize(){
		return m_size_inbytes;
	}

};

}

#include "GpuData.inc"


#endif /* INCLUDE_GPU_GPUDATA_H_ */
