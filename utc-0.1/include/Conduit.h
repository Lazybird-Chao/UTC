#ifndef UTC_CONDUIT_H_
#define UTC_CONDUIT_H_

#include "UtcBasics.h"
#include "TaskBase.h"

#include <vector>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cassert>

namespace iUtc
{

class Conduit{

public:
	Conduit();

	Conduit(TaskBase* srctask, TaskBase* dsttask);

	void setCapacity();
	int getCapacity();

	int Write(void* DataPtr, int DataSize, int DataTag);
	int WriteBy(ThreadRank wthread, void* DataPtr, int DataSize, int DataTag);
	int WriteByAll(void* DataPtr, int DataSize, int DataTag);

	int Read(void* DataPtr, int DataSize, int DataTag);
	int ReadBy(ThreadRank wthread, void* DataPtr, int DataSize, int DataTag);
	int ReadByAll(void* DataPtr, int DataSize, int DataTag);



private:
	Task<T1>* m_srcTask;
	Task<T2>* m_dstTask;
	TaskId m_srcId;
	TaskId m_dstId;

	int m_conduitId;
	int m_capacity;

	int m_availableMsgBuffHead;
	int m_availableMsgBuffTail;

	std::map<MessageKey, void*> m_msgBuffPool;
	std::mutex m_hasMsgBuffMutex;
	std::condition_variable m_hasMsgBuffCond;






};



}// namespace iUtc




#endif

