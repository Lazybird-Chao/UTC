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

	// actually not bond to obj instance
	void rstoreFence();
	void rstoreQuiet();

	void waitChange(T value);
	void waitChangeUntil(condCMPType cond, T value);

	// for m_size = 1
	operator T() const;
	T& operator =(T value);

private:
	long m_size;
	int m_typesize;
	metaDataType m_datatype;
	T* m_dataPtr;
	UserTaskBase* m_userTaskObj;
	int m_currentPE;

	std::mutex* m_ctxMutex;
	SpinLock* m_ctxSpinMutex;




};

}// end namespace iUtc

#include "GlobalScopedData.inc"

#endif /* UTC_GBLOBALSCOPEDDATA_H_ */
