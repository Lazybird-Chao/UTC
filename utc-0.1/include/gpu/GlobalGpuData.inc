/*
 * GlobalGpuData.inc
 *
 *  Created on: Sep 14, 2017
 *      Author: chaoliu
 */



#include "cuda_runtime.h"
#include "TaskUtilities.h"
#include "TaskBase.h"




template <typename T>
GlobalGpuData<T>::GlobalGpuData(unsigned long size, MemType memtype ){
	m_dim = 1;
	m_size = size;
	m_size_inbytes = m_size*sizeof(T);
	isinitialized = iscleaned = false;
	m_hostPtr = m_devicePtr = NULL;
	m_memtype = memtype;

	m_worldToGroup = NULL;
	m_win_comm = NULL;

}

template<typename T>
GlobalGpuData<T>::~GlobalGpuData(){
	m_worldToGroup = NULL;
	m_win_comm = NULL;
	if(isinitialized == true && iscleaned == false){
		destroy();
	}
}

template<typename T>
int GlobalGpuData<T>::init(){
	m_objMutex.lock();
	if(isinitialized == false && iscleaned == false){
		m_worldToGroup = getCurrentTask()->getProcWorldToTaskGroupMap();
		m_win_comm = getCurrentTask()->getTaskMpiWindow()->get_scoped_win_comm();

		if(getCurrentUtcGpuCtx()->getCurrentDeviceAttr(cudaDevAttrManagedMemory) ==1){
			//std::cout<<"using cuda managed for umem"<<std::endl;
			checkCudaRuntimeErrors(cudaMallocManaged(&m_devicePtr, m_size_inbytes));
			m_hostPtr = m_devicePtr;
		}
		else{
			std::cout<<"GlobalGpuData need cuda unified memory support!!!"<<std::endl;
			m_hostPtr = m_devicePtr = NULL;
		}

		MPI_Info win_info = MPI_INFO_NULL;
		MPI_Info_create(&win_info);
		MPI_Info_set(win_info, "same_size", "true");
		int rc = MPI_Win_create(m_hostPtr,
								m_size_inbytes,
								1,
								win_info,
								*m_win_comm,
								&m_win);

		if (rc!=MPI_SUCCESS) {
			char errmsg[MPI_MAX_ERROR_STRING];
			int errlen;
			MPI_Error_string(rc, errmsg, &errlen);
			printf("MPI_Win_allocate_shared error message = %s\n",errmsg);
			//MPI_Abort(*scoped_win_comm, rc);
		}

		 MPI_Win_lock_all(MPI_MODE_NOCHECK /* use 0 instead if things break */,
				 m_win);
		 MPI_Info_free(&win_info);
		 MPI_Barrier(*m_win_comm);

		isinitialized = true;
	}
	m_objMutex.unlock();
	return 0;
}

template<typename T>
int GlobalGpuData<T>::destroy(){
	m_objMutex.lock();
	if(isinitialized == true && iscleaned == false){
		MPI_Barrier(*m_win_comm);
		MPI_Win_unlock_all(m_win);
		MPI_Win_free(&m_win);

		if(m_devicePtr)
			checkCudaRuntimeErrors(cudaFree(m_devicePtr));
		iscleaned = true;
	}
	m_objMutex.unlock();
}

template<typename T>
int GlobalGpuData<T>::rload(T* dst, int remotePE){
	//int remotePE = m_worldToGroup->at(remotePE);
	m_objMutex.lock();
	if(m_size_inbytes < (1<<31))
		MPI_Get(dst, m_size_inbytes,
				remotePE, 0, m_size_inbytes, MPI_CHAR,
				m_win);
	else
		MPI_Get(dst, m_size_inbytes,
						remotePE, 0, m_size_inbytes/4, MPI_INT,
						m_win);

	MPI_Win_flush_local(remotePE, m_win);
	m_objMutex.unlock();
	return 0;
}

template<typename T>
int GlobalGpuData<T>::rstore(T* src, int remotePE){
	//int remotePE = m_worldToGroup->at(remotePE);
	m_objMutex.lock();
	if(m_size_inbytes < (1<<31))
		MPI_Put(src, m_size_inbytes,
				remotePE, 0, m_size_inbytes, MPI_CHAR,
				m_win);
	else
		MPI_Put(src, m_size_inbytes,
						remotePE, 0, m_size_inbytes/4, MPI_INT,
						m_win);

	MPI_Win_flush_local(remotePE, m_win);
	m_objMutex.unlock();
	return 0;
}

template<typename T>
void GlobalGpuData<T>::fence(){
	m_objMutex.lock();
	MPI_Win_flush_all(m_win);
	m_objMutex.unlock();
	return;
}




