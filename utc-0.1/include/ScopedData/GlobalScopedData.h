/*
 * GblobalScopedData.h
 *
 *  Created on: Jun 24, 2016
 *      Author: chaoliu
 */

#ifndef UTC_GBLOBALSCOPEDDATA_H_
#define UTC_GBLOBALSCOPEDDATA_H_

#include "GlobalScopedDataBase.h"
#include "SharedDataLock.h"
#include "SpinLock.h"
#include "UserTaskBase.h"
#include "FastMutex.h"

#include <mutex>
#include "shmem.h"

namespace iUtc{

enum class metaDataType{
	unknown=-1,
	_char,
	_short,
	_int,
	_long,
	_float,
	_double,
	_longlong,
	_longdouble
};

enum class condCMPType{
	unknown=-1,
	_eq = SHMEM_CMP_EQ,
	_ne = SHMEM_CMP_NE,
	_gt = SHMEM_CMP_GT,
	_le = SHMEM_CMP_LE,
	_lt = SHMEM_CMP_LT,
	_ge = SHMEM_CMP_GE
};

template <typename T>
class GlobalScopedData: public GlobalScopedDataBase{
public:
	GlobalScopedData();

	GlobalScopedData(long size);

	GlobalScopedData(UserTaskBase* userTaskObj);

	GlobalScopedData(UserTaskBase* userTaskObj, long size);

	~GlobalScopedData();

	void init();
	void destroy();

	T* getPtr();
	// only for different process on same node, else not work.
	T* rgetPtr(int remotePE);

	int getSize();

	int getCurrentPE();

	T load(int index=0);
	void store(T value, int index=0);

	T rload(int remotePE, int index=0);
	void rstore(int remotePE, T value, int index=0);

	int loadblock(T* dst, int startIdx, int blocks);
	int storeblock(T* src, int startIdx, int blocks);

	int rloadblock(int remotePE, T* dst, int startIdx, int blocks);
	int rstoreblock(int remotePE, T* src, int startIdx, int blocks);

	/* the openSHMEM fence/quiet only ensure the order of remote put ops
	 * that happend in the calling PE, but not indicate if the data is
	 * already writ to remote memory, only sh_barrier ensure the completion of
	 * remote memory update. However, sh_barrier is hard to use in our multithread
	 * context, especially if there are several multithreads task runnint in one process,
	 * the sh_barrier of different task may cause problem(pollute the barrier match procedure).
	 *
	 * The MPI barrier or other collective communication ops could work, because of
	 * the MPI communicator. We set different communicator for different task. Noting that
	 * this kind of communicator use in multithreads environment in our UTC is still not
	 * full tested. Maybe, there are still problems.
	 *
	 * We need a well thread-friend MPI/openSHMEM implementation.
	 *
	 * Right now some network middleware is actually thread safe and thread friendly.
	 * For the best, we should use this network middle ware directly to implement
	 * data communication. This will need much more work. Redesign and implement Conduit.
	 *
	 *
	 * So when doing a remote put, remote PE don't know when the data are written
	 * to its memory and can be used. Remote PE can use a flag to do wait operation
	 * to see if the write is completed.
	 *
	 * Be careful of this.
	 */
	void rstoreFence();
	void rstoreQuiet();

	void waitChange(T value);
	void waitChangeUntil(condCMPType cond, T value);

	/* the caller PE of remote put call setflag(), the remote PE call waitflag()
	 * then the remote PE knows the data come from rput is ready for use.
	 * this two ops should be used as pair, else will cause error for success
	 * using.
	 */
	void rstoreSetFinishFlag(int remotePE);
	void rstoreWaitFinishFlag(int rstoreCaller);

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
	UserTaskBase* m_userTaskObj;
	int m_currentPE;

	int *m_flagOldValue;
	int *m_flagValue;

	//std::mutex* m_ctxMutex;
	FastMutex *m_ctxMutex;
	SpinLock* m_ctxSpinMutex;

	//std::mutex m_objMutex;
	FastMutex m_objMutex;


};

}// end namespace iUtc

#include "GlobalScopedData.inc"

#endif /* UTC_GBLOBALSCOPEDDATA_H_ */
