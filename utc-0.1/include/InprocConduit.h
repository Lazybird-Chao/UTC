#ifndef UTC_INPROC_CONDUIT_H_
#define UTC_INPROC_CONDUIT_H_

#include "TaskBase.h"
#include "ConduitBase.h"

#include <vector>
#include <map>
#include <mutex>
#include <condition_variable>
#include <fstream>


namespace iUtc
{


/*enum OpType{
	unknown =0,
	read,
	write,
	readby,
	writeby
};*/

class InprocConduit: public ConduitBase
{

	struct BuffInfo
	{
		void *dataPtr = nullptr;
		void *bufferPtr = nullptr;
		int buffIdx = -1;
		int callingWriteThreadCount =0;
		int callingReadThreadCount = 0;
		int buffSize = 0;
		int reduceBuffsizeSensor = 0;

	};

public:
	//InprocConduit();

	//InprocConduit(TaskBase* srctask, TaskBase* dsttask);

	InprocConduit(TaskBase* srctask, TaskBase* dsttask, int cdtId, int capacity);

	//void setCapacity(int capacity);
	//int getCapacity();

	//std::string getName();

	/*TaskBase* getSrcTask();
    TaskBase* getDstTask();
    TaskBase* getAnotherTask();
    void Connect(TaskBase* src, TaskBase* dst);
    ConduitId getConduitId();*/


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


	//
	~InprocConduit();
	void clear();

private:
	//
	void initInprocConduit();
	/*void checkOnSameProc(TaskBase* src, TaskBase* dst);

	ConduitManager* m_cdtMgr;*/
	//
	TaskBase* m_srcTask;
	TaskBase* m_dstTask;
	TaskId m_srcId;
	TaskId m_dstId;

	//std::string m_Name;
	int m_conduitId;
	int m_capacity;
	int m_noFinishedOpCapacity;  // different meaning as m_capacity!!!

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
	BuffInfo* m_srcAvailableBuff;
	std::map<MessageTag, BuffInfo*> m_srcBuffPool;
	std::map<MessageTag, int> m_srcBuffPoolWaitlist;
	// use this idx to refer each buffer's access mutex and buffwritten flag
	std::vector<int> m_srcBuffIdx;
	std::vector<std::mutex> m_srcBuffAccessMutex;
	// signal that thread copied data into buffer
	std::vector<std::condition_variable> m_srcBuffDataWrittenCond;
	// signal that thread copied data outof buffer
	std::vector<std::condition_variable> m_srcBuffDataReadCond;
	std::vector<int> m_srcBuffDataWrittenFlag;
	std::vector<int> m_srcBuffDataReadFlag;
	// used to control buffer allocate and update buffer related info
	std::mutex m_srcBuffManagerMutex;
	// src thread wait for this cond when buff full, dst thread notify this cond when
	// release a buff
	std::condition_variable m_srcBuffAvailableCond;
	//src thread notify this cond when insert a new buff item to pool in write, dst thread
	//wait this cond when can't find request message in pool in read
	std::condition_variable m_srcNewBuffInsertedCond;


	std::mutex m_srcOpCheckMutex;
	int *m_srcOpRotateCounter;
	int *m_srcOpRotateCounterIdx;
	int *m_srcOpRotateFinishFlag;
	std::condition_variable m_srcOpFinishCond;
	std::condition_variable m_srcAvailableNoFinishedOpCond;
	int m_srcAvailableNoFinishedOpCount;



	int m_dstAvailableBuffCount;
	BuffInfo* m_dstAvailableBuff;
    std::map<MessageTag, BuffInfo*> m_dstBuffPool;
    std::map<MessageTag, int> m_dstBuffPoolWaitlist;
    std::vector<int> m_dstBuffIdx;
    std::vector<std::mutex> m_dstBuffAccessMutex;
    std::vector<std::condition_variable> m_dstBuffDataWrittenCond;
    std::vector<std::condition_variable> m_dstBuffDataReadCond;
    std::vector<int> m_dstBuffDataWrittenFlag;
    std::vector<int> m_dstBuffDataReadFlag;
    std::mutex m_dstBuffManagerMutex;
    std::condition_variable m_dstBuffAvailableCond;
    std::condition_variable m_dstNewBuffInsertedCond;


    std::mutex m_dstOpCheckMutex;
    int *m_dstOpRotateCounter;
    int *m_dstOpRotateCounterIdx;
    int *m_dstOpRotateFinishFlag;
    std::condition_variable m_dstOpFinishCond;
    std::condition_variable m_dstAvailableNoFinishedOpCond;
    int m_dstAvailableNoFinishedOpCount;

    /*used by writeby and readby to set a flag for check and waiting
    as only asigned thread do the op, other threads will go one their process,
    use this to make sure all threads know the data transfer is complete,
    safe to use the source data or dst data.
    an item is released from this set by calling readby_finish(), so if this is
    not called, the corespond item will not be erased from the finishset.*/
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
    InprocConduit& operator=(const InprocConduit &other)=delete;
    InprocConduit(const InprocConduit &other)=delete;

};



}// namespace iUtc




#endif

