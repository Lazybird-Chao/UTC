#ifndef UTC_INPROC_CONDUIT_H_
#define UTC_INPROC_CONDUIT_H_

#include "TaskBase.h"
#include "ConduitBase.h"
#include "LockFreeRingbufferQueue.h"


#include <vector>
#include <deque>
#include <map>
#include <mutex>
#include <condition_variable>
#include <future>
#include <fstream>
#include <atomic>
#include "boost/thread/latch.hpp"

namespace iUtc{

class InprocConduit: public ConduitBase{
	
public:

	struct MsgInfo{
		void *dataPtr = nullptr;
		void *smallDataBuff = nullptr;
		DataSize_t dataSize = 0;
		bool usingPtr = false;
		int msgTag = -1;
		std::atomic<int> *safeRelease=nullptr;
	};
	typedef struct MsgInfo	MsgInfo_t;

	InprocConduit(TaskBase* src, TaskBase* dst, int cdtId);

	/*
	 *
	 */
	int Write(void* DataPtr, DataSize_t DataSize, int tag);
    int WriteBy(ThreadRank_t thread, void* DataPtr, DataSize_t DataSize, int tag);
    void WriteBy_Finish(int tag);
	
	/*
	 *
	 */
	int BWrite(void* DataPtr, DataSize_t DataSize, int tag);
	int BWriteBy(ThreadRank_t thread, void* DataPtr, DataSize_t DataSize, int tag);
	void BWriteBy_Finish(int tag);

	/*
	 *
	 */
	int PWrite(void* DataPtr, DataSize_t DataSize, int tag);
    int PWriteBy(ThreadRank_t thread, void* DataPtr, DataSize_t DataSize, int tag);
    void PWriteBy_Finish(int tag);

    int Read(void* DataPtr, DataSize_t DataSize, int tag);
	int ReadBy(ThreadRank_t thread, void* DataPtr, DataSize_t DataSize, int tag);
	void ReadBy_Finish(int tag);

	/*
	 *
	 */
	int AsyncWrite(void* DataPtr, DataSize_t DataSize, int tag);
	void AsyncWrite_Finish(int tag);


	int AsyncRead(void* DataPtr, DataSize_t DataSize, int tag);
	void AsyncRead_Finish(int tag);


	/*
	 *
	 */
	~InprocConduit();

private:
	void initInprocConduit();

	void clear();


	/*
	 *
	 */
	 TaskBase *m_srcTask;
	 TaskBase *m_dstTask;
	 TaskId_t  m_srcId;
	 TaskId_t  m_dstId;

	 int m_conduitId;
	 int m_numSrcLocalThreads;
	 int m_numDstLocalThreads;

	 /*
	  * src side buffer pool, this is used for src writing and dst reading.
	  *
	  * each msg should have a unique tag, in this way several src threads
	  * can insert msg to pool at same time, or else, it's hard to decide the 
	  * order of inserted mag of same tag by different threads, and then recv 
	  * can't distinguish which msg he is receving.
	  * if we don't require unique tag for each msg, we sould assure the msg 
	  * inserted order is same as the programmed order. So if several src threads
	  * do insert at same time, we at list need to block other thread and let only
	  * one thread do insert(although memcpy may do at same time, but pool pos taken
	  * only allow one thread at one time).
	  */
	LockFreeQueue<MsgInfo_t,INPROC_CONDUIT_CAPACITY_DEFAULT>	*m_srcBuffQueue;
	LockFreeQueue<MsgInfo_t, INPROC_CONDUIT_CAPACITY_DEFAULT>	*m_srcInnerMsgQueue;
	std::map<int, MsgInfo_t*>	m_srcBuffMap;
	/*
	 * using token-ring-net mechanism. Start from thread local rank 0, 
	 * each thread responds for doing the r/w one by one.
	 * inside boost::latch, it use a pair of condition variable and mutex
	 * to do this synchization.
	 * each thread has a tokenflag to indicate whoes turn to do w/r. 
	 */
	std::vector<boost::latch*>	m_srcOpThreadLatch;
	int 	*m_srcOpTokenFlag;
	/*
	 * when transfer msg using address, writer need wait reader finish
	 * copy data and then return. So using this atomic flag for reader 
	 * to notify writer the data copy is finished.
	 */
	std::atomic<int> 	*m_srcUsingPtrFinishFlag;


	/*
	 * dst side buffer pool
	 */
	LockFreeQueue<MsgInfo_t,INPROC_CONDUIT_CAPACITY_DEFAULT>	*m_dstBuffQueue;
	LockFreeQueue<MsgInfo_t, INPROC_CONDUIT_CAPACITY_DEFAULT> 	*m_dstInnerMsgQueue;
	std::map<int, MsgInfo_t*>	m_dstBuffMap; 
	//
	std::vector<boost::latch*>	m_dstOpThreadLatch;
	int 	*m_dstOpTokenFlag;
	//
	std::atomic<int> 	*m_dstUsingPtrFinishFlag;


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


	/*
	 * For async op
	 */
	struct AsyncWorkArgs
	{
		void* DataPtr = nullptr;
		DataSize_t DataSize =0;
		int tag = -1;
		int WorkType = 0;  // 1: read, 2: write
	};
	std::map<int, std::promise<void>> m_srcAsyncReadFinishSet;
	std::map<int, std::promise<void>> m_srcAsyncWriteFinishSet;
	bool m_srcNewAsyncWork;
	bool m_srcAsyncWorkerCloseSig;
	bool m_srcAsyncWorkerOn;
	std::deque<AsyncWorkArgs> m_srcAsyncWorkQueue;
	std::condition_variable m_srcNewAsyncWorkCond;
	std::mutex m_srcNewAsyncWorkMutex;

	std::map<int, std::promise<void>> m_dstAsyncReadFinishSet;
	std::map<int, std::promise<void>> m_dstAsyncWriteFinishSet;
	bool m_dstNewAsyncWork;
	bool m_dstAsyncWorkerCloseSig;
	bool m_dstAsyncWorkerOn;
	std::deque<AsyncWorkArgs> m_dstAsyncWorkQueue;
	std::condition_variable m_dstNewAsyncWorkCond;
	std::mutex m_dstNewAsyncWorkMutex;


    // the max time period in second that reader wait for writer transferring data
    int TIME_OUT = 100;
    // the max time period in microsecods that an async worker wait for workload
    int ASYNC_TIME_OUT = 3000;

    // output debug log to specific file
    static thread_local std::ofstream *m_threadOstream;


	//
	InprocConduit& operator=(const InprocConduit &other)=delete;
    InprocConduit(const InprocConduit &other)=delete;

};



}






#endif


