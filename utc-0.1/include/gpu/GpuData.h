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
	int clean();
	bool iscleaned;
	bool isinitialized;

public:
	typedef T _Type;

	GpuData(MemType memtype = MemType::pageable);
	GpuData(unsigned long size, MemType memtype = MemType::pageable);
	GpuData(unsigned long size_x, unsigned long size_y, MemType memtype = MemType::pageable);
	GpuData(unsigned long size_x, unsigned long size_y, unsigned long size_z, MemType memtype = MemType::pageable);

	int initMem(unsigned long size_x, MemType memtype= MemType::pageable);
	int initMem(unsigned long size_x, unsigned long size_y, MemType memtype= MemType::pageable);
	int initMem(unsigned long size_x, unsigned long size_y, unsigned long size_z, MemType memtype= MemType::pageable);

	void cleanMem();

	~GpuData();

	/*
	 * get memory address
	 */
	T *get(){
		if(m_memtype == MemType::unified ||
				m_status == MemStatus::host)
			return m_hostPtr;
		else{
			return m_devicePtr;
		}
	}
	T *getH(){
		return m_hostPtr;
	}
	T *getD(){
		return m_devicePtr;
	}

	T *get(bool isModefied){
		if(m_memtype == MemType::unified ||
				m_status == MemStatus::host){
			m_status = MemStatus::host;
			return m_hostPtr;
		}
		else{
			m_status = MemStatus::device;
			return m_devicePtr;
		}

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
		if(m_memtype == MemType::unified){
			m_status = MemStatus::synced;
			return;
		}
		checkCudaRuntimeErrors(cudaMemcpy(m_devicePtr, m_hostPtr,
				m_size_inbytes, cudaMemcpyHostToDevice));
		m_status = MemStatus::synced;
	}

	void syncD(){
		if(m_memtype == MemType::unified){
			m_status = MemStatus::synced;
			return;
		}
		checkCudaRuntimeErrors(cudaMemcpy(m_hostPtr, m_devicePtr,
				m_size_inbytes, cudaMemcpyDeviceToHost));
		m_status = MemStatus::synced;
	}

	void sync(){
		if(m_memtype == MemType::unified){
			m_status = MemStatus::synced;
			return;
		}
		if(m_status == MemStatus::host)
			checkCudaRuntimeErrors(cudaMemcpy(m_devicePtr, m_hostPtr,
						m_size_inbytes, cudaMemcpyHostToDevice));
		else if(m_status == MemStatus::device)
			checkCudaRuntimeErrors(cudaMemcpy(m_hostPtr, m_devicePtr,
						m_size_inbytes, cudaMemcpyDeviceToHost));
		m_status = MemStatus::synced;
	}

	void syncH(unsigned long offset, unsigned long size){
		if(m_memtype == MemType::unified){
			m_status = MemStatus::synced;
			return;
		}
		checkCudaRuntimeErrors(cudaMemcpy(&m_devicePtr[offset], &m_hostPtr[offset],
				size, cudaMemcpyHostToDevice));
		m_status = MemStatus::synced;
	}

	void syncD(unsigned long offset, unsigned long size){
		if(m_memtype == MemType::unified){
			m_status = MemStatus::synced;
			return;
		}
		checkCudaRuntimeErrors(cudaMemcpy(&m_hostPtr[offset], &m_devicePtr[offset],
				size, cudaMemcpyDeviceToHost));
		m_status = MemStatus::synced;
	}

	void sync(unsigned long offset, unsigned long size){
		if(m_memtype == MemType::unified){
			m_status = MemStatus::synced;
			return;
		}
		if(m_status == MemStatus::host)
			checkCudaRuntimeErrors(cudaMemcpy(&m_devicePtr[offset], &m_hostPtr[offset],
						size, cudaMemcpyHostToDevice));
		else if(m_status == MemStatus::device)
			checkCudaRuntimeErrors(cudaMemcpy(&m_hostPtr[offset], &m_devicePtr[offset],
						size, cudaMemcpyDeviceToHost));
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
		m_status = MemStatus::host;
	}

	/*
	 * put to specified memory
	 */
	void putH(const T *src){
		memcpy(m_hostPtr, src, m_size_inbytes);
		m_status = MemStatus::host;
	}
	void putD(const T *src){
		checkCudaRuntimeErrors(cudaMemcpy(m_devicePtr, src,
				m_size_inbytes, cudaMemcpyDefault));
		m_status = MemStatus::device;
	}
	void put(const T *src){
		memcpy(m_hostPtr, src, m_size_inbytes);
		if(m_memtype != MemType::unified){
			checkCudaRuntimeErrors(cudaMemcpy(m_devicePtr, src,
							m_size_inbytes, cudaMemcpyDefault));
		}
		m_status = MemStatus::synced;
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
		m_status = MemStatus::host;
	}
	void initD(int value=0){
		checkCudaRuntimeErrors(cudaMemset(m_devicePtr, value, m_size_inbytes));
		m_status = MemStatus::device;
	}
	void init(int value =0){
		memset(m_hostPtr, value, m_size_inbytes);
		if(m_memtype != MemType::unified){
			checkCudaRuntimeErrors(cudaMemset(m_devicePtr, value, m_size_inbytes));
		}
		m_status = MemStatus::synced;
	}

	void initH(const T* other){
		memcpy(m_hostPtr, other, m_size_inbytes);
		m_status = MemStatus::host;
	}
	void initD(const T* other){
		checkCudaRuntimeErrors(cudaMemcpy(m_devicePtr, other,
				m_size_inbytes, cudaMemcpyDefault));
		m_status = MemStatus::device;
	}
	void init(const T* other){
		memcpy(m_hostPtr, other, m_size_inbytes);
		if(m_memtype != MemType::unified){
			checkCudaRuntimeErrors(cudaMemcpy(m_devicePtr, other,
							m_size_inbytes, cudaMemcpyDefault));
		}
		m_status = MemStatus::synced;
	}

	/* dest should be a host pointer */
	void fetchH(T* dest){
		memcpy(dest, m_hostPtr, m_size_inbytes);
	}
	void fetchD(T* dest){
		checkCudaRuntimeErrors(cudaMemcpy(dest, m_devicePtr,
				m_size_inbytes, cudaMemcpyDefault));
	}
	void fetch(T* dest){
		if(m_memtype == MemType::unified ||
				m_status == MemStatus::host)
			memcpy(dest, m_hostPtr, m_size_inbytes);
		else{
			checkCudaRuntimeErrors(cudaMemcpy(m_hostPtr, m_devicePtr,
						m_size_inbytes, cudaMemcpyDefault));
			memcpy(dest, m_hostPtr, m_size_inbytes);
			m_status = MemStatus::synced;
		}
	}


	/* dest should be a device pointer*/
	void moveH(T* dest){
		checkCudaRuntimeErrors(cudaMemcpy(dest, m_hostPtr,
				m_size_inbytes, cudaMemcpyDefault));
	}
	void moveD(T* dest){
		checkCudaRuntimeErrors(cudaMemcpy(dest, m_devicePtr,
				m_size_inbytes, cudaMemcpyDefault));
	}
	void move(T* dest){
		if(m_memtype == MemType::unified ||
				m_status == MemStatus::device)
			checkCudaRuntimeErrors(cudaMemcpy(dest, m_devicePtr,
					m_size_inbytes, cudaMemcpyDefault));
		else{
			checkCudaRuntimeErrors(cudaMemcpy(m_devicePtr, m_hostPtr,
								m_size_inbytes, cudaMemcpyDefault));
			checkCudaRuntimeErrors(cudaMemcpy(dest, m_devicePtr,
								m_size_inbytes, cudaMemcpyDefault));
			m_status = MemStatus::synced;
		}
	}






};

}

#include "GpuData.inc"


#endif /* INCLUDE_GPU_GPUDATA_H_ */
