/*
 * GlobalScopedData_internal.inc
 *
 *  Created on: Sep 11, 2017
 *      Author: Chao
 */



/*
 * this kind of data should be task object related data, not each task thread related,
 * a such data is shared by all threads on same process;
 *
 * any thread of same process can call following functions, but we'd better keep that
 * one time only one thread call some method to get/put data, especially remote
 * accessing methods. One thread finish (call fence or quiet), then other thread can
 * call, otherwise may mess up(not sure...)
 * Actually, as this data object is shared by all threads on process, we only need
 * always use some designated thread to deal with this data obj-related remote
 * memory access.
 *
 */
namespace iUtc{


template <typename T>
class GlobalScopedData: public GlobalScopedDataBase{
public:
	GlobalScopedData(long size=1);

	int init();
	int destroy();

	~GlobalScopedData();

	T* getPtr();

	int getSize();

	/* return the process rank in the task-related communicator not global */
	int getCurrentPE();

	T load(int index=0);
	void store(T value, int index=0);

	T rload(int remotePE, int index=0);
	void rstore(int remotePE, T value, int index=0);

	int loadblock(T* dst, int startIdx, int blocks);
	int storeblock(T* src, int startIdx, int blocks);

	/*
	 * call proc will do remote get/put data from the 'remotePE' proc
	 * they are non-blocking calls, so after call we don't know if
	 * data transfer is completed.
	 */
	int rloadblock(int remotePE, T* dst, int startIdx, int blocks);
	int rstoreblock(int remotePE, T* src, int startIdx, int blocks);

	/*
	 * fence and quiet are not collective calls, so only the one who call rload, rstore
	 * need call them to ensure the rload/rstore complete
	 */
	void fence();
	void quiet();
	/*
	 * barrier is collective call, should be called by all processes of this task
	 */
	void barrier();

	// for m_size = 1, only for local data, not remote
	operator T() const;
	T& operator =(T value);
	//
	T& operator[] (long index);

private:
	long m_size;
	int m_typesize;
	metaDataType m_datatype;
	T* m_dataPtr;
	internal_MPIWin *m_taskMpiWindow;

	UserTaskBase* m_userTaskObj;
	int m_globalPrank;
	int m_taskPrank;
	std::map<int, int> *m_worldToGroup;

	//std::mutex m_objMutex;
	FastMutex m_objMutex;

	//
	bool initialized;
	bool destroied;

};

}//end namespace

namespace iUtc{

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
	m_userTaskObj = NULL;

	initialized = false;
	destroied = false;

	/*
	 * these things need to be done after task threads are launched, as
	 * we use some data( stored in task thread local data etc.) that are set
	 * after task threads launched. so we add separate init destroy methods
	 * and do it explicitly in user task implementation codes
	 */
	/*
	m_globalPrank = getCurrentTask()->getCurrentProcRank();
	m_worldToGroup = getCurrentTask()->getProcWorldToTaskGroupMap();
	m_taskPrank = m_worldToGroup->at(m_globalPrank);
	*/

	//m_taskMpiWindow = getCurrentTask()->getTaskMpiWindow();
	/* the task mpiwindow init/finalize procedures are done in usertaskbase's
	 * preInit() and preExit()
	 */
	//scoped_shmem_init_comm(*m_taskMpiWindow);
	//m_dataPtr = (T*)scoped_shmem_malloc(m_size*m_typesize, *m_taskMpiWindow);

	m_globalPrank = -1;
	m_worldToGroup = NULL;
	m_taskPrank = -1;
	m_taskMpiWindow = NULL;
	m_dataPtr = NULL;

}


template<typename T>
int GlobalScopedData<T>::init(){
	m_objMutex.lock();
	if( !initialized && !destroied){
		m_globalPrank = getCurrentTask()->getCurrentProcRank();
		m_worldToGroup = getCurrentTask()->getProcWorldToTaskGroupMap();
		m_taskPrank = m_worldToGroup->at(m_globalPrank);

		m_taskMpiWindow = getCurrentTask()->getTaskMpiWindow();
		m_dataPtr = (T*)scoped_shmem_malloc(m_size*m_typesize, *m_taskMpiWindow);
		initialized = true;
		destroied = false;

#ifdef SHOW_DEBUG
	std::cout<<ERROR_LINE<<"global scoped data initialized."
			<<m_globalPrank<<" "<<m_taskPrank<<std::endl;
#endif
	}
	m_objMutex.unlock();
	return 0;
}

template<typename T>
int GlobalScopedData<T>::destroy(){
	m_objMutex.lock();
	if(initialized && !destroied){
		if(m_dataPtr){
			//std::cout<<ERROR_LINE<<std::endl;
			scoped_shmem_free(m_dataPtr, *m_taskMpiWindow);
		}
		destroied = true;
		m_dataPtr = nullptr;
	}
	m_objMutex.unlock();
	return 0;
}

template<typename T>
GlobalScopedData<T>::~GlobalScopedData(){
#ifdef SHOW_DEBUG
	std::cout<<ERROR_LINE<<"global scoped data do destructor." <<std::endl;
#endif
	if(initialized && !destroied){
		if(m_dataPtr){
			scoped_shmem_free(m_dataPtr, *m_taskMpiWindow);
		}
		destroied = true;
	}
}

template<typename T>
T* GlobalScopedData<T>::getPtr(){
	if( !initialized  || destroied){
		std::cerr<<"Error: global data not initialized or destroied."<<std::endl;
		return 0;
	}
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
	if( !initialized  || destroied){
		std::cerr<<"Error: global data not initialized or destroied."<<std::endl;
		return (T)0;
	}
	return m_dataPtr[index];
}

template<typename T>
void GlobalScopedData<T>::store(T value, int index){
	if( !initialized  || destroied){
		std::cerr<<"Error: global data not initialized or destroied."<<std::endl;
	}
	m_dataPtr[index] = value;
}

template<typename T>
T GlobalScopedData<T>::rload(int remotePE, int index){
	if( !initialized  || destroied){
		std::cerr<<"Error: global data not initialized or destroied."<<std::endl;
		return (T)0;
	}
	T tmpdata;
	//remotePE = m_worldToGroup->at(remotePE);
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
	if( !initialized  || destroied){
		std::cerr<<"Error: global data not initialized or destroied."<<std::endl;
	}
	//remotePE = m_worldToGroup->at(remotePE);
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
	if( !initialized  || destroied){
		std::cerr<<"Error: global data not initialized or destroied."<<std::endl;
		return 0;
	}
	memcpy(dst, &m_dataPtr[startIdx], blocks*m_typesize);
	return 0;
}

template<typename T>
int GlobalScopedData<T>::storeblock(T* src, int startIdx, int blocks){
	if( !initialized  || destroied){
		std::cerr<<"Error: global data not initialized or destroied."<<std::endl;
		return 0;
	}
	memcpy(&m_dataPtr[startIdx], src, blocks*m_typesize);
	return 0;
}

template<typename T>
int GlobalScopedData<T>::rloadblock(int remotePE, T* dst, int startIdx, int blocks){
	if( !initialized  || destroied){
		std::cerr<<"Error: global data not initialized or destroied."<<std::endl;
		return 0;
	}
	//remotePE = m_worldToGroup->at(remotePE);
	m_objMutex.lock();
	switch(m_datatype){
	case metaDataType::_char:
		scoped_shmem_char_get((char*)dst, (char*)&(m_dataPtr[startIdx]), blocks, remotePE, *m_taskMpiWindow);
		break;
	case metaDataType::_short:
		scoped_shmem_short_get((short*)dst, (short*)&(m_dataPtr[startIdx]), blocks, remotePE, *m_taskMpiWindow);
		break;
	case metaDataType::_int:
		scoped_shmem_int_get((int*)dst, (int*)&(m_dataPtr[startIdx]), blocks, remotePE, *m_taskMpiWindow);
		break;
	case metaDataType::_long:
		scoped_shmem_long_get((long*)dst, (long*)&(m_dataPtr[startIdx]), blocks, remotePE, *m_taskMpiWindow);
		break;
	case metaDataType::_float:
		scoped_shmem_float_get((float*)dst, (float*)&(m_dataPtr[startIdx]), blocks, remotePE, *m_taskMpiWindow);
		break;
	case metaDataType::_double:
		scoped_shmem_double_get((double*)dst, (double*)&(m_dataPtr[startIdx]), blocks, remotePE, *m_taskMpiWindow);
		break;
	default:
		std::cout<<"Error, unsupported datatype in rstore()!!!"<<std::endl;
		break;
	}

	//scoped_shmem_getmem((void*)dst, (void*)&(m_dataPtr[startIdx]), blocks*m_typesize, remotePE, *m_taskMpiWindow);

	m_objMutex.unlock();
	return 0;
}

template<typename T>
int GlobalScopedData<T>::rstoreblock(int remotePE, T* src, int startIdx, int blocks){
	if( !initialized  || destroied){
		std::cerr<<"Error: global data not initialized or destroied."<<std::endl;
		return 0;
	}
	//remotePE = m_worldToGroup->at(remotePE);
	m_objMutex.lock();
	switch(m_datatype){
	case metaDataType::_char:
		scoped_shmem_char_put((char*)&(m_dataPtr[startIdx]), (char*)src, blocks, remotePE, *m_taskMpiWindow);
		break;
	case metaDataType::_short:
		scoped_shmem_short_put((short*)&(m_dataPtr[startIdx]), (short*)src, blocks, remotePE, *m_taskMpiWindow);
		break;
	case metaDataType::_int:
		scoped_shmem_int_put((int*)&(m_dataPtr[startIdx]), (int*)src, blocks, remotePE, *m_taskMpiWindow);
		break;
	case metaDataType::_long:
		scoped_shmem_long_put((long*)&(m_dataPtr[startIdx]), (long*)src, blocks, remotePE, *m_taskMpiWindow);
		break;
	case metaDataType::_float:
		scoped_shmem_float_put((float*)&(m_dataPtr[startIdx]), (float*)src, blocks, remotePE, *m_taskMpiWindow);
		break;
	case metaDataType::_double:
		scoped_shmem_double_put((double*)&(m_dataPtr[startIdx]), (double*)src, blocks, remotePE, *m_taskMpiWindow);
		break;
	default:
		std::cout<<"Error, unsupported datatype in rstore()!!!"<<std::endl;
		break;
	}
	//scoped_shmem_putmem((void*)&(m_dataPtr[startIdx]), (void*)src, blocks*m_typesize, remotePE, *m_taskMpiWindow);
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



}// end namespace



