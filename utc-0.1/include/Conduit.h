#ifndef UTC_CONDUIT_H_
#define UTC_CONDUIT_H_

#include "UtcBasics.h"
#include "TaskBase.h"
//#include "ConduitManager.h"

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

//
class ConduitManager;

class Conduit{

	struct BuffInfo
	{
		void *dataPtr = nullptr;
		int buffIdx = 0;
		int callingWriteThreadCount =0;
		int callingReadThreadCount = 0;
		bool safeReleaseAfterRead = false;
	};

public:
	Conduit();

	Conduit(TaskBase* srctask, TaskBase* dsttask);

	Conduit(TaskBase* srctask, TaskBase* dsttask, int capacity);

	//void setCapacity(int capacity);
	int getCapacity();

	std::string getName();

	TaskBase* getSrcTask();
    TaskBase* getDstTask();
    void Connect(TaskBase* src, TaskBase* dst);
    ConduitId getConduitId();

    /* semantic: A task write the message in DataPtr to conduit inner buffer
     * 			 when it returns, the data is buffed in conduit, and sender
     *			 can modify this sent data safely.
     * thread-ops(in one process):
     *	   Write()      executed by all task threads, but only one thread do the data transfer
     *	   WriteBy()    the arg specified thread do the transfer
     *     WriteByAll()  all threads do the transfer, so DataPrt and DataTag should be
     *					 different, otherwise case conflict and behave not ensured correct
	*/
	int Write(void* DataPtr, int DataSize, int tag);
	int WriteBy(ThreadRank wthread, void* DataPtr, int DataSize, int tag);
	int WriteByAll(void* DataPtr, int DataSize, int tag);
	/*
	 * With same semantic and multithread behave as Write()
	 */
	int Read(void* DataPtr, int DataSize, int tag);
	int ReadBy(ThreadRank wthread, void* DataPtr, int DataSize, int tag);
	int ReadByAll(void* DataPtr, int DataSize, int tag);



	~Conduit();
	void clear();

private:
	//
	void initConduit();
	void checkOnSameProc(TaskBase* src, TaskBase* dst);

	ConduitManager* m_cdtMgr;
	//
	TaskBase* m_srcTask;
	TaskBase* m_dstTask;
	TaskId m_srcId;
	TaskId m_dstId;

	std::string m_Name;
	int m_conduitId;
	int m_capacity;

	int m_numSrcLocalThreads;
	int m_numDstLocalThreads;

	 /*Here we use two buffer pool, one for direction src->dst
	 one for dst->src. In this way src and dst can do their read/write
	 separately, like a full-dupliex channel.
	 the bufferpool's manage mutex and each buffer's RW mutex are shared
	 by reader and writer.
	 src write to src-buffer read from dst-buffer
	 dst write to dst-buffer read from src-buffer
	 */
	int m_srcAvailableBuffCount;
	std::map<MessageTag, BuffInfo*> m_srcBuffPool;
	// use this idx to refer each buffer's access mutex and buffwritten flag
	std::vector<int> m_srcBuffIdx;
	std::vector<std::mutex> m_srcBuffAccessMutex;
	std::vector<std::condition_variable> m_srcBuffWcallbyAllCond;
	std::vector<int> m_srcBuffWrittenFlag;
	// used to control buffer allocate and release
	std::mutex m_srcBuffManagerMutex;
	// src thread wait for this cond when buff full, dst thread notify this cond when
	// release a buff
	std::condition_variable m_srcBuffAvailableCond;
	// src thread notify this cond when insert a new buff item to pool in write, dst thread
	// wait this cond when can't find request message in pool in read
	std::condition_variable m_srcNewBuffInsertedCond;
	// used for one thread to block other thread doing write or read
	std::mutex m_srcHoldOtherthreadsWriteMutex;
	std::mutex m_srcHoldOtherthreadsReadMutex;


	int m_dstAvailableBuffCount;
    std::map<MessageTag, BuffInfo*> m_dstBuffPool;
    std::vector<int> m_dstBuffIdx;
    std::vector<std::mutex> m_dstBuffAccessMutex;
    std::vector<std::condition_variable> m_dstBuffWcallbyAllCond;
    std::vector<int> m_dstBuffWrittenFlag;
    std::mutex m_dstBuffManagerMutex;
    std::condition_variable m_dstBuffAvailableCond;
    std::condition_variable m_dstNewBuffInsertedCond;
    std::mutex m_dstHoldOtherthreadsWriteMutex;
    std::mutex m_dstHoldOtherthreadsReadMutex;

    // the max time period in second that reader wait for writer transferring data
    int TIME_OUT = 100;
    // output debug log to specific file
    static thread_local std::ofstream *m_threadOstream;


    //
    Conduit& operator=(const Conduit &other)=delete;
    Conduit(const Conduit &other)=delete;

};



}// namespace iUtc




#endif

