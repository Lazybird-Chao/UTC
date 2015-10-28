#ifndef UTC_CONDUIT_H_
#define UTC_CONDUIT_H_

#include "UtcBasics.h"
#include "TaskBase.h"
//#include "ConduitManager.h"

#include <vector>
#include <map>
#include <mutex>
#include <condition_variable>
#include <fstream>


namespace iUtc
{

//
class ConduitManager;

enum OpType{
	unknown =0,
	read,
	write,
	readby,
	writeby
};
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
    TaskBase* getAnotherTask();
    void Connect(TaskBase* src, TaskBase* dst);
    ConduitId getConduitId();

    /* semantic: A task write the message in DataPtr to conduit inner buffer
     * 			 when it returns, the data is buffed in conduit, and sender
     *			 can modify this sent data safely.
     * thread-ops(in one process):
     *	   Write()      executed by all task threads, but only one thread do the data transfer
     *	   WriteBy()    the arg specified thread do the transfer
     *	   WriteBy_Finish()   check and wait finishing WriteBy() operation of a message
	*/
	int Write(void* DataPtr, int DataSize, int tag);
	int WriteBy(ThreadRank thread, void* DataPtr, int DataSize, int tag);
	void WriteBy_Finish(int tag);
	/*
	 * With same semantic and multithread behave as Write()
	 */
	int Read(void* DataPtr, int DataSize, int tag);
	int ReadBy(ThreadRank thread, void* DataPtr, int DataSize, int tag);
	void ReadBy_Finish(int tag);


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
	// used to control buffer allocate and update buffer related info
	std::mutex m_srcBuffManagerMutex;
	// src thread wait for this cond when buff full, dst thread notify this cond when
	// release a buff
	std::condition_variable m_srcBuffAvailableCond;
	// src thread notify this cond when insert a new buff item to pool in write, dst thread
	// wait this cond when can't find request message in pool in read
	std::condition_variable m_srcNewBuffInsertedCond;
	// used for one thread to check if need do the real write op
	std::mutex m_srcWriteOpCheckMutex;
	std::mutex m_srcReadOpCheckMutex;
	// each counter is used to record how many threads called current write op
	// for all read ops in program, each read op will use a new counter, because at most there
	// are capacity+1 read ops active at same time, so we can use capacity+1 counters repeately
	int *m_srcWriteOpRotateCounter;
	int *m_srcReadOpRotateCounter;
	// thread i use idx[i] to remember which counter for use
	int *m_srcWriteOpRotateCounterIdx;
	int *m_srcReadOpRotateCounterIdx;
	//  each counter has a flag to indicate that real write op is finished
	int *m_srcWriteOpRotateFinishFlag;
	int *m_srcReadOpRotateFinishFlag;
	//  used by late coming thread to wait the real write op finish, and first coming thread will
	//  use for notify
	std::condition_variable m_srcWriteOpFinishCond;
	std::condition_variable m_srcReadOpFinishCond;


	int m_dstAvailableBuffCount;
    std::map<MessageTag, BuffInfo*> m_dstBuffPool;
    std::vector<int> m_dstBuffIdx;
    std::vector<std::mutex> m_dstBuffAccessMutex;
    std::vector<std::condition_variable> m_dstBuffWcallbyAllCond;
    std::vector<int> m_dstBuffWrittenFlag;
    std::mutex m_dstBuffManagerMutex;
    std::condition_variable m_dstBuffAvailableCond;
    std::condition_variable m_dstNewBuffInsertedCond;
    std::mutex m_dstWriteOpCheckMutex;
    int *m_dstWriteOpRotateCounter;
    int *m_dstWriteOpRotateCounterIdx;
    int *m_dstWriteOpRotateFinishFlag;
    std::condition_variable m_dstWriteOpFinishCond;
    std::mutex m_dstReadOpCheckMutex;
    int *m_dstReadOpRotateCounter;
    int *m_dstReadOpRotateCounterIdx;
    int *m_dstReadOpRotateFinishFlag;
    std::condition_variable m_dstReadOpFinishCond;


    // used by writeby and readby to set a flag for check and waiting
    // as only asigned thread do the op, other threads will go one their process,
    // use this to make sure all threads know the data transfer is complete,
    // safe to use the source data or dst data.
    std::map<int, int> m_readbyFinishSet;
    std::mutex m_readbyFinishMutex;
    std::condition_variable m_readbyFinishCond;
    std::map<int, int> m_writebyFinishSet;
	std::mutex m_writebyFinishMutex;
	std::condition_variable m_writebyFinishCond;



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

