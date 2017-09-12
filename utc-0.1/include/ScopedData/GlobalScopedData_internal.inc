/*
 * GlobalScopedData_internal.inc
 *
 *  Created on: Sep 11, 2017
 *      Author: Chao
 */

#ifndef INCLUDE_SCOPEDDATA_GLOBALSCOPEDDATA_INTERNAL_INC_
#define INCLUDE_SCOPEDDATA_GLOBALSCOPEDDATA_INTERNAL_INC_
#include "TaskUtilities.h"
#include "TaskBase.h"

namespace iUtc{

template<typename T>
GlobalScopedData<T>::GlobalScopedData(UserTaskBase* userTaskObj, long size)
:GlobalScopedDataBase(){
	m_size = size;
	m_typesize = sizeof(T);
	if(typeid(T)==typeid(char))
		m_datatype = metaDataType::_char;
	else if(typeid(T)==typeid(short))
		m_datatype = metaDataType::_short;
	else if(typeid(T)==typeid(int))
		m_datatype = metaDataType::_int;
	else if(typeid(T)==typeid(long))
		m_datatype = metaDataType::_long;
	else if(typeid(T)==typeid(float))
		m_datatype = metaDataType::_float;
	else if(typeid(T)==typeid(double))
		m_datatype = metaDataType::_double;
	else if(typeid(T)==typeid(long long))
		m_datatype = metaDataType::_longlong;
	else if(typeid(T)==typeid(long double))
		m_datatype = metaDataType::_longdouble;
	else
		m_datatype = metaDataType::unknown;
	m_userTaskObj = userTaskObj;

	m_globalPrank = userTaskObj->__processId;
	m_taskPrank = userTaskObj->__worldRankTranslate->at(m_globalPrank);
	m_worldToGroup = *(userTaskObj->__worldRankTranslate);

	m_taskMpiWindow = getCurrentTask()->getTaskMpiWindow();
	/* the task mpiwindow init/finalize procedures are done in usertaskbase's
	 * preInit() and preExit()
	 */
	//scoped_shmem_init_comm(*m_taskMpiWindow);
	m_dataPtr = (T*)scoped_shmem_malloc(m_size*m_typesize, *m_taskMpiWindow);

}


template<typename T>
GlobalScopedData<T>::~GlobalScopedData(){
	if(m_dataPtr){
		scoped_shmem_free(m_dataPtr, *m_taskMpiWindow);
	}
}

template<typename T>
T* GlobalScopedData<T>::getPtr(){
	return (T*)m_dataPtr;
}

template<typename T>
int GlobalScopedData<T>::getSize(){
	return m_size;
}

template<typename T>
int GlobalScopedData<T>::getCurrentPE(){
	return m_taskPrank;
}

template<typename T>
T GlobalScopedData<T>::load(int index){
	return m_dataPtr[index];
}

template<typename T>
void GlobalScopedData<T>::store(T value, int index){
	m_dataPtr[index] = value;
}

template<typename T>
T GlobalScopedData<T>::rload(int remotePE, int index){
	T tmpdata;
	remotePE = m_worldToGroup[remotePE];
	m_objMutex.lock();
	switch(m_datatype){
		case metaDataType::_char:
			tmpdata = scoped_shmem_char_g((char*)&m_dataPtr[index], remotePE, *m_taskMpiWindow);
			break;
		case metaDataType::_short:
			tmpdata = scoped_shmem_short_g((short*)&m_dataPtr[index], remotePE, *m_taskMpiWindow);
			break;
		case metaDataType::_int:
			tmpdata = scoped_shmem_int_g((int*)&m_dataPtr[index], remotePE, *m_taskMpiWindow);
			break;
		case metaDataType::_long:
			tmpdata = scoped_shmem_long_g((long*)&m_dataPtr[index], remotePE, *m_taskMpiWindow);
			break;
		case metaDataType::_float:
			tmpdata = scoped_shmem_float_g((float*)&m_dataPtr[index], remotePE, *m_taskMpiWindow);
			break;
		case metaDataType::_double:
			tmpdata = scoped_shmem_double_g((double*)&m_dataPtr[index], remotePE, *m_taskMpiWindow);
			break;
		default:
			std::cout<<"Error, unsupported datatype in rload()!!!"<<std::endl;
			break;
	}
	m_objMutex.unlock();
	return tmpdata;
}

template<typename T>
void GlobalScopedData<T>::rstore(int remotePE, T value, int index){
	remotePE = m_worldToGroup[remotePE];
	m_objMutex.lock();
	switch(m_datatype){
		case metaDataType::_char:
			scoped_shmem_char_p((char*)&m_dataPtr[index], value, remotePE, *m_taskMpiWindow);
			break;
		case metaDataType::_short:
			scoped_shmem_short_p((short*)&m_dataPtr[index], value, remotePE, *m_taskMpiWindow);
			break;
		case metaDataType::_int:
			scoped_shmem_int_p((int*)&m_dataPtr[index], value, remotePE, *m_taskMpiWindow);
			break;
		case metaDataType::_long:
			scoped_shmem_long_p((long*)&m_dataPtr[index], value, remotePE, *m_taskMpiWindow);
			break;
		case metaDataType::_float:
			scoped_shmem_float_p((float*)&m_dataPtr[index], value, remotePE, *m_taskMpiWindow);
			break;
		case metaDataType::_double:
			scoped_shmem_double_p((double*)&m_dataPtr[index], value, remotePE, *m_taskMpiWindow);
			break;
		default:
			std::cout<<"Error, unsupported datatype in rstore()!!!"<<std::endl;
			break;
	}
	m_objMutex.unlock();
	return;
}

template<typename T>
int GlobalScopedData<T>::loadblock(T* dst, int startIdx, int blocks){
	memcpy(dst, &m_dataPtr[startIdx], blocks*m_typesize);
	return 0;
}

template<typename T>
int GlobalScopedData<T>::storeblock(T* src, int startIdx, int blocks){
	memcpy(&m_dataPtr[startIdx], src, blocks*m_typesize);
	return 0;
}

template<typename T>
int GlobalScopedData<T>::rloadblock(int remotePE, T* dst, int startIdx, int blocks){
	remotePE = m_worldToGroup[remotePE];
	m_objMutex.lock();
	scoped_shmem_getmem((void*)dst, (void*)&(m_dataPtr[startIdx]), blocks*m_typesize, remotePE, *m_taskMpiWindow);
	m_objMutex.unlock();
	return 0;
}

template<typename T>
int GlobalScopedData<T>::rstoreblock(int remotePE, T* src, int startIdx, int blocks){
	remotePE = m_worldToGroup[remotePE];
	m_objMutex.lock();
	scoped_shmem_putmem((void*)&(m_dataPtr[startIdx]), (void*)src, blocks*m_typesize, remotePE, *m_taskMpiWindow);
	m_objMutex.unlock();
	return 0;
}

template<typename T>
void GlobalScopedData<T>::fence(){
	m_objMutex.lock();
	scoped_shmem_fence(*m_taskMpiWindow);
	m_objMutex.unlock();
}

template<typename T>
void GlobalScopedData<T>::quiet(){
	m_objMutex.lock();
	scoped_shmem_quiet(*m_taskMpiWindow);
	m_objMutex.unlock();
}

template<typename T>
void GlobalScopedData<T>::barrier(){
	m_objMutex.lock();
	scoped_shmem_barrier(*m_taskMpiWindow);
	m_objMutex.unlock();
}


template<typename T>
GlobalScopedData<T>::operator T() const{
	return m_dataPtr[0];
}

template<typename T>
T& GlobalScopedData<T>::operator=(T value){
	m_dataPtr[0]= value;
	return m_dataPtr[0];
}

template<typename T>
T& GlobalScopedData<T>::operator [](long index){
	return m_dataPtr[index];
}



}



#endif /* INCLUDE_SCOPEDDATA_GLOBALSCOPEDDATA_INTERNAL_INC_ */
