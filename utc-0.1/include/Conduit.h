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

	void setCapacity(int capacity);
	int getCapacity();

	TaskBase* getSrcTask();
    TaskBase* getDstTask();
    void Connect(TaskBase* src, TaskBase* dst);
    ConduitId getConduitId();

	int Write(void* DataPtr, int DataSize, int DataTag);
	int WriteBy(ThreadRank wthread, void* DataPtr, int DataSize, int DataTag);
	int WriteByAll(void* DataPtr, int DataSize, int DataTag);

	int Read(void* DataPtr, int DataSize, int DataTag);
	int ReadBy(ThreadRank wthread, void* DataPtr, int DataSize, int DataTag);
	int ReadByAll(void* DataPtr, int DataSize, int DataTag);





private:
	//
	TaskBase* m_srcTask;
	TaskBase* m_dstTask;
	TaskId m_srcId;
	TaskId m_dstId;

	int m_conduitId;
	int m_capacity;

	//
	int m_srcAvailableMsgBuffHead;
	int m_srcAvailableMsgBuffTail;

	std::map<MessageTag, void*> m_srcMsgBuffPool;
	std::mutex m_srcHasMsgBuffMutex;
	std::condition_variable m_srcHasMsgBuffCond;

	int m_dstAvailableMsgBuffHead;
    int m_dstAvailableMsgBuffTail;

    std::map<MessageTag, void*> m_dstMsgBuffPool;
    std::mutex m_dstHasMsgBuffMutex;
    std::condition_variable m_dstHasMsgBuffCond;






};



}// namespace iUtc




#endif

