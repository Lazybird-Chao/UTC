/*
 * GlobalGpuData.h
 *
 *  Created on: Sep 14, 2017
 *      Author: chaoliu
 */

#ifndef GLOBALGPUDATA_H_
#define GLOBALGPUDATA_H_

#include "FastMutex.h"

#include "UtcGpuBasics.h"
#include "helper_cuda.h"

#include <mpi.h>

namespace iUtc{

enum class MemType;

template<typename T>
class GlobalGpuData{
private:
	unsigned long m_size;
	unsigned long m_size_inbytes;
	int m_dim;
	MemType m_memtype;

	bool isinitialized;
	bool iscleaned;

	T *m_hostPtr;
	T *m_devicePtr;

	MPI_Comm *m_win_comm;
	MPI_Win m_win;
	std::map<int, int> *m_worldToGroup;

	FastMutex m_objMutex;

public:
	int init();
	int destroy();

	GlobalGpuData(unsigned long size, MemType memtype = MemType::unified);
	~GlobalGpuData();

	T *get(){
		return m_hostPtr;
	}

	int rload(T* dst, int PE);
	int rstore(T* src, int PE);

	void fence();

};

#include "GlobalGpuData.inc"

}




#endif /* GLOBALGPUDATA_H_ */
