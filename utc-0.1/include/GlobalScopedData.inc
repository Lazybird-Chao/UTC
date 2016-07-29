/*
 * GlobalScopedData.inc
 *
 *  Created on: Jun 24, 2016
 *      Author: chaoliu
 */
#include "UtcContext.h"

#include <typeinfo>

namespace iUtc{

template<typename T>
GlobalScopedData<T>::GlobalScopedData()
:GlobalScopedDataBase(){
	m_size = 1;
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
	m_userTaskObj = nullptr;
	m_currentPE = -1;
	m_dataPtr = nullptr;
	init();
}

template<typename T>
GlobalScopedData<T>::GlobalScopedData(long size)
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
	m_userTaskObj = nullptr;
	m_currentPE = -1;
	m_dataPtr = nullptr;
	init();
}

template<typename T>
GlobalScopedData<T>::GlobalScopedData(UserTaskBase* userTaskObj)
:GlobalScopedDataBase(){
	m_size = 1;
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
	m_currentPE = -1;
	m_dataPtr = nullptr;
	init();
}

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
	m_currentPE = -1;
	m_dataPtr = nullptr;
	init();
}


template<typename T>
GlobalScopedData<T>::~GlobalScopedData(){
	destroy();
}

template<typename T>
void GlobalScopedData<T>::init(){
	UtcContext &ctx = UtcContext::getContext();
	m_currentPE = ctx.getProcRank();
	m_ctxMutex = ctx.getCtxMutex();
	m_ctxSpinMutex = ctx.getCtxSpinMutex();
	//m_currentPE = shmem_my_pe();
	//m_dataPtr = (T*)shmem_malloc(m_size*m_typesize);
	m_dataPtr = (T*)shmalloc(m_size*m_typesize);

	m_flagValue= (int*)shmalloc(sizeof(int)*ctx.numProcs());
	m_flagOldValue=(int*)shmalloc(sizeof(int)*ctx.numProcs());;
	for(int i=0; i<ctx.numProcs(); i++){
		m_flagValue[i]=0;
		m_flagOldValue[i]=0;
	}

}

template<typename T>
void GlobalScopedData<T>::destroy(){
	if(m_dataPtr){
		//shmem_free((void*)m_dataPtr);
		shfree((void*)m_dataPtr);
		shfree((void*)m_flagValue);
		shfree((void*)m_flagOldValue);
	}

	m_ctxMutex = nullptr;
	m_ctxSpinMutex = nullptr;
}

template<typename T>
T* GlobalScopedData<T>::getPtr(){
	return (T*)m_dataPtr;
}

template<typename T>
T* GlobalScopedData<T>::rgetPtr(int remotePE){
	if(!shmem_addr_accessible(m_dataPtr, remotePE ))
		return nullptr;
	return (T*)shmem_ptr(m_dataPtr, remotePE);
}

template<typename T>
int GlobalScopedData<T>::getSize(){
	return m_size;
}

template<typename T>
int GlobalScopedData<T>::getCurrentPE(){
	return m_currentPE;
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
	m_ctxSpinMutex->lock();
	switch(m_datatype){
		case metaDataType::_char:
			tmpdata = shmem_char_g((char*)&m_dataPtr[index], remotePE);
			break;
		case metaDataType::_short:
			tmpdata = shmem_short_g((short*)&m_dataPtr[index], remotePE);
			break;
		case metaDataType::_int:
			tmpdata = shmem_int_g((int*)&m_dataPtr[index], remotePE);
			break;
		case metaDataType::_long:
			tmpdata = shmem_long_g((long*)&m_dataPtr[index], remotePE);
			break;
		case metaDataType::_float:
			tmpdata = shmem_float_g((float*)&m_dataPtr[index], remotePE);
			break;
		case metaDataType::_double:
			tmpdata = shmem_double_g((double*)&m_dataPtr[index], remotePE);
			break;
		default:
			std::cout<<"Error, unsupported datatype in rload()!!!"<<std::endl;
			break;
	}
	m_ctxSpinMutex->unlock();
	return tmpdata;
}

template<typename T>
void GlobalScopedData<T>::rstore(int remotePE, T value, int index){
	m_ctxSpinMutex->lock();
	switch(m_datatype){
		case metaDataType::_char:
			shmem_char_p((char*)&m_dataPtr[index], value, remotePE);
			break;
		case metaDataType::_short:
			shmem_short_p((short*)&m_dataPtr[index], value, remotePE);
			break;
		case metaDataType::_int:
			shmem_int_p((int*)&m_dataPtr[index], value, remotePE);
			break;
		case metaDataType::_long:
			shmem_long_p((long*)&m_dataPtr[index], value, remotePE);
			break;
		case metaDataType::_float:
			 shmem_float_p((float*)&m_dataPtr[index], value, remotePE);
			break;
		case metaDataType::_double:
			shmem_double_p((double*)&m_dataPtr[index], value, remotePE);
			break;
		default:
			std::cout<<"Error, unsupported datatype in rstore()!!!"<<std::endl;
			break;
	}
	m_ctxSpinMutex->unlock();
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
	m_ctxMutex->lock();
	/*switch(m_datatype){
		case metaDataType::_char:
			shmem_char_get(dst, &m_dataPtr[startIdx], blocks, remotePE);
			break;
		case metaDataType::_short:
			shmem_short_get(dst, &m_dataPtr[startIdx], blocks, remotePE);
			break;
		case metaDataType::_int:
			shmem_int_get(dst, &m_dataPtr[startIdx], blocks, remotePE);
			break;
		case metaDataType::_long:
			shmem_long_get(dst, &m_dataPtr[startIdx], blocks, remotePE);
			break;
		case metaDataType::_float:
			shmem_float_get(dst, &m_dataPtr[startIdx], blocks, remotePE);
			break;
		case metaDataType::_double:
			shmem_double_get(dst, &m_dataPtr[startIdx], blocks, remotePE);
			break;
		default:
			break;
	}*/
	shmem_getmem((void*)dst, (void*)&(m_dataPtr[startIdx]), blocks*m_typesize, remotePE);
	m_ctxMutex->unlock();
}

template<typename T>
int GlobalScopedData<T>::rstoreblock(int remotePE, T* src, int startIdx, int blocks){
	m_ctxMutex->lock();
	/*switch(m_datatype){
		case metaDataType::_char:
			shmem_char_put(&m_dataPtr[startIdx], src, blocks, remotePE);
			break;
		case metaDataType::_short:
			shmem_short_put(&m_dataPtr[startIdx], src, blocks, remotePE);
			break;
		case metaDataType::_int:
			shmem_int_put(&m_dataPtr[startIdx], src, blocks, remotePE);
			break;
		case metaDataType::_long:
			shmem_long_put(&m_dataPtr[startIdx], src, blocks, remotePE);
			break;
		case metaDataType::_float:
			shmem_float_put(&m_dataPtr[startIdx], src, blocks, remotePE);
			break;
		case metaDataType::_double:
			shmem_double_put(&m_dataPtr[startIdx], src, blocks, remotePE);
			break;
		default:
			break;
	}*/
	shmem_putmem((void*)&(m_dataPtr[startIdx]), (void*)src, blocks*m_typesize, remotePE);
	m_ctxMutex->unlock();
}

template<typename T>
void GlobalScopedData<T>::rstoreFence(){
	m_ctxMutex->lock();
	shmem_fence();
	m_ctxMutex->unlock();
}

template<typename T>
void GlobalScopedData<T>::rstoreQuiet(){
	m_ctxMutex->lock();
	shmem_quiet();
	m_ctxMutex->unlock();
}

template<typename T>
void GlobalScopedData<T>::waitChange(T value){
	m_objMutex.lock();
	switch(m_datatype){
	case metaDataType::_short:
		shmem_short_wait(m_dataPtr, value);
		break;
	case metaDataType::_int:
		shmem_int_wait(m_dataPtr, value);
		break;
	case metaDataType::_long:
		shmem_long_wait(m_dataPtr, value);
		break;
	default:
		std::cout<<"Error, unsupported datatype in waitChange()!!!"<<std::endl;
		break;
	}
	m_objMutex.unlock();
}

template<typename T>
void GlobalScopedData<T>::waitChangeUntil(condCMPType cond, T value){
	m_objMutex.lock();
	switch(m_datatype){
	case metaDataType::_short:
		shmem_short_wait_until(m_dataPtr, cond, value);
		break;
	case metaDataType::_int:
		shmem_int_wait_until(m_dataPtr, cond, value);
		break;
	case metaDataType::_long:
		shmem_long_wait_until(m_dataPtr, cond, value);
		break;
	default:
		std::cout<<"Error, unsupported datatype in waitChangeUntil()!!!"<<std::endl;
		break;
	}
	m_objMutex.unlock();
}

template<typename T>
void GlobalScopedData<T>::rstoreSetFinishFlag(int remotePE){
	m_ctxSpinMutex->lock();
	//shmem_fence();
	//shmem_int_add(&m_flagValue[m_currentPE], 1, remotePE);
	int oldvalue = shmem_int_g(&m_flagValue[m_currentPE], remotePE);
	shmem_int_p(&m_flagValue[m_currentPE], oldvalue+1, remotePE);
	m_ctxSpinMutex->unlock();
}

template<typename T>
void GlobalScopedData<T>::rstoreWaitFinishFlag(int rstoreCaller){
	m_objMutex.lock();
	if(m_flagOldValue[rstoreCaller] == m_flagValue[rstoreCaller])
		shmem_int_wait(&m_flagValue[rstoreCaller], m_flagOldValue[rstoreCaller]);
#ifdef USE_DEBUG_ASSERT
	assert(m_flagValue[rstoreCaller]==m_flagOldValue[rstoreCaller]+1);
#endif
	m_flagOldValue[rstoreCaller] = m_flagValue[rstoreCaller];
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


}//end namespace iUtc
