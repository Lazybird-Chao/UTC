/*
 * GlobalScopedData_internal.h
 *
 *  Created on: Sep 11, 2017
 *      Author: Chao
 */

#ifndef INCLUDE_SCOPEDDATA_GLOBALSCOPEDDATA_INTERNAL_H_
#define INCLUDE_SCOPEDDATA_GLOBALSCOPEDDATA_INTERNAL_H_
#include <map>


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
template <typename T>
class GlobalScopedData: public GlobalScopedDataBase{
public:
	GlobalScopedData(long size=1);

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
	std::map<int, int> &m_worldToGroup;

	//std::mutex m_objMutex;
	FastMutex m_objMutex;

};




#endif /* INCLUDE_SCOPEDDATA_GLOBALSCOPEDDATA_INTERNAL_H_ */
