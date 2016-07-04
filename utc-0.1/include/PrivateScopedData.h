/*
 * ScopedData.h
 *
 *  Created on: Jun 16, 2016
 *      Author: chaoliu
 */

#ifndef UTC_PRIVATE_SCOPEDDATA_H_
#define UTC_PRIVATE_SCOPEDDATA_H_

#include "PrivateScopedDataBase.h"
#include "UserTaskBase.h"

#include <atomic>
#include <iostream>
#include "boost/thread/tss.hpp"

namespace iUtc{

template <typename T>
class PrivateScopedData: public PrivateScopedDataBase{
public:
	PrivateScopedData();

	PrivateScopedData(long size);

	PrivateScopedData(UserTaskBase* userTaskObj);

	PrivateScopedData(UserTaskBase* userTaskObj, long size);

	~PrivateScopedData();

	void init();
	void destroy();

	T* getPtr();

	int getSize();

	T load(int index=0);
	void store(T value, int index=0);

	int loadblock(T* dst, int startIdx, int blocks);
	int storeblock(T* src, int startIdx, int blocks);

	// these only support when m_size = 1;
	operator T() const;
	T& operator =(T value);



private:
	long m_size;
	int  m_typesize;
	std::atomic<int> m_numThreads;
	boost::thread_specific_ptr<T> m_dataPtr;
	UserTaskBase* m_userTaskObj;


};


}// end namespace iUtc

#include "PrivateScopedData.inc"


#endif /* UTC_SCOPEDDATA_H_ */
