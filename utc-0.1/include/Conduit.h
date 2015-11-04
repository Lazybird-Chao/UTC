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
		bool isBuffered = true;
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


    /* Blocking & Buffered write
     *
     * Blocking op: when method returns, the write operation is finished,
     *              data has been written to conduit, safe to use data.
     * Buffered op: there is conduit inner buffer created to store the data,
     *              the writer doesn't need to wait for reader starting read
     *              the data.
     *
     * thread-ops(in one process):
     *	   Write()      executed by all task threads, but only one thread do the data transfer
     *	   WriteBy()    the arg specified thread do the transfer
     *	   WriteBy_Finish()   check and wait finishing WriteBy() operation of a message
	*/
	int BWrite(void* DataPtr, int DataSize, int tag);
	int BWriteBy(ThreadRank thread, void* DataPtr, int DataSize, int tag);
	void BWriteBy_Finish(int tag);

	/* Blocking & Buffered when needed write
	 *
	 * For this write operation, whether to buffer the message data is decided
	 * by system. When write happens, if the corresponding read is already
	 * wait for the message, then it will not buffer the data, reader will
	 * read the data directly. Otherwise, data will be stored in conduit
	 * internal buffer like BWrite() does.
	 */
	int Write(void* DataPtr, int DataSize, int tag);
    int WriteBy(ThreadRank thread, void* DataPtr, int DataSize, int tag);
    void WriteBy_Finish(int tag);


    /* Blocking & Non-buffered write
     *
     * There is no conduit internal buffer created to store data,
     * it pass "DataPtr" the address to reader.
     * But it will not return until reader copied data away.
     */
    int PWrite(void* DataPtr, int DataSize, int tag);
    int PWriteBy(ThreadRank thread, void* DataPtr, int DataSize, int tag);
    void PWriteBy_Finish(int tag);



	/*
	 * Blocking op: when returns, the read operation is finished, reader get
	 *              the data, and is safe to use.
	 * thread-ops(in one process): same as write() operation.
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
	std::map<MessageTag, int> m_srcBuffPoolWaitlist;
	/*std::condition_variable m_srcBuffPoolWaitlistCond;*/
	// use this idx to refer each buffer's access mutex and buffwritten flag
	std::vector<int> m_srcBuffIdx;
	std::vector<std::mutex> m_srcBuffAccessMutex;
	// signal that thread copied data into buffer
	std::vector<std::condition_variable> m_srcBuffDataWrittenCond;
	// signal that thread copied data outof buffer
	std::vector<std::condition_variable> m_srcBuffDataReadCond;
	std::vector<int> m_srcBuffDataWrittenFlag;
	std::vector<int> m_srcBuffDataReadFlag;
	// when all write threads finish write, use this to signal reader to release buffer,
	// changer this var from every buffer's to one, may cause more compete use
	std::condition_variable m_srcBuffSafeReleaseCond;
	// used to control buffer allocate and update buffer related info
	std::mutex m_srcBuffManagerMutex;
	// src thread wait for this cond when buff full, dst thread notify this cond when
	// release a buff
	std::condition_variable m_srcBuffAvailableCond;
	//src thread notify this cond when insert a new buff item to pool in write, dst thread
	//wait this cond when can't find request message in pool in read
	std::condition_variable m_srcNewBuffInsertedCond;
	// used for thread to check if need do the real write op
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
    std::map<MessageTag, int> m_dstBuffPoolWaitlist;
   /* std::condition_variable m_dstBuffPoolWaitlistCond;*/
    std::vector<int> m_dstBuffIdx;
    std::vector<std::mutex> m_dstBuffAccessMutex;
    std::vector<std::condition_variable> m_dstBuffDataWrittenCond;
    std::vector<std::condition_variable> m_dstBuffDataReadCond;
    std::vector<int> m_dstBuffDataWrittenFlag;
    std::vector<int> m_dstBuffDataReadFlag;
    std::condition_variable m_dstBuffSafeReleaseCond;
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
    // an item is released from this set by calling readby_finish(), so if this is
    // not called, the corespond item will not be erased from the finishset.
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

