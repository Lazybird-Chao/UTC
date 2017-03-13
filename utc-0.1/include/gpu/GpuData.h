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

	int m_status; //

public:
	typedef T _Type;

	GpuData(int size, MemType memtype = MemType::pagable);
	GpuData(int size_x, int size_y, MemType memtype = MemType::pagable);
	GpuData(int size_x, int size_y, int size_z, MemType memtype = MemType::pagable);

	~GpuData();

	T *getH(){
		return m_hostPtr;
	}
	T *getD(){
		return m_devicePtr;
	}

	T* getH(size_t offset){
		return &m_hostPtr[offset];
	}
	T* getD(size_t offset){
		return &m_devicePtr[offset];
	}

	T at(int index){
		return m_hostPtr[index];
	}
	void put(size_t index, const T value){
		m_hostPtr[index] = value;
	}




};

}

#include "GpuData.inc"


#endif /* INCLUDE_GPU_GPUDATA_H_ */
