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
#include <cstdint>
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

	InprocConduit(TaskBase* src, TaskBase* dst, int cdtId, std::string name);

	/*
	 *
	 */
	int Write(void* DataPtr, DataSize_t DataSize, int tag);
    int WriteBy(ThreadRank_t thread, void* DataPtr, DataSize_t DataSize, int tag);
    void WriteBy_Finish(int tag){};
    int WriteByFirst(void* DataPtr, DataSize_t DataSize, int tag);
	
	/*
	 *
	 */
	int BWrite(void* DataPtr, DataSize_t DataSize, int tag);
	int BWriteBy(ThreadRank_t thread, void* DataPtr, DataSize_t DataSize, int tag);
	void BWriteBy_Finish(int tag){};

	/*
	 *
	 */
	int PWrite(void* DataPtr, DataSize_t DataSize, int tag);
    int PWriteBy(ThreadRank_t thread, void* DataPtr, DataSize_t DataSize, int tag);
    void PWriteBy_Finish(int tag){};

    int Read(void* DataPtr, DataSize_t DataSize, int tag);
	int ReadBy(ThreadRank_t thread, void* DataPtr, DataSize_t DataSize, int tag);
	void ReadBy_Finish(int tag){};
	int ReadByFirst(void *DataPtr, DataSize_t DataSize, int tag);

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

	void asyncWorkerImpl(int myTaskid);

	int threadWriteImpl(void* DataPtr, DataSize_t DataSize, int tag, int myTaskid);

	int threadReadImpl(void* DataPtr, DataSize_t DataSize, int tag, int myTaskid);


	/*
	 *
	 */
	 TaskBase *m_srcTask;
	 TaskBase *m_dstTask;
	 TaskId_t  m_srcId;
	 TaskId_t  m_dstId;

	 std::string m_Name;
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
	/*std::vector<boost::latch*>	m_srcOpThreadLatch;*/
	int 	*m_srcOpTokenFlag;
	// for small msg, do not use latch, use atomic as fake latch
	/*std::atomic<int> *m_srcOpThreadAtomic;*/
	/*
	 * when transfer msg using address, writer need wait reader finish
	 * copy data and then return. So using this atomic flag for reader 
	 * to notify writer the data copy is finished.
	 */
	std::atomic<int> 	*m_srcUsingPtrFinishFlag;

	/*
	 * new method for multi threads do op and let one do, others wait
	 */
	std::vector<std::atomic<int>*> m_srcOpThreadAvailable;
	std::vector<std::atomic<int>*> m_srcOpThreadFinish;
	std::vector<boost::latch*> m_srcOpThreadFinishLatch;
	std::vector<std::atomic<int>*> m_dstOpThreadAvailable;
	std::vector<std::atomic<int>*> m_dstOpThreadFinish;
	std::vector<boost::latch*> m_dstOpThreadFinishLatch;
	int m_nOps2=32;

	std::atomic<int> *m_srcOpThreadIsFirst;
	int *m_srcOpFirstIdx;
	std::atomic<int> *m_dstOpThreadIsFirst;
	int *m_dstOpFirstIdx;
	int m_nOps=32;

	/*
	 * dst side buffer pool
	 */
	LockFreeQueue<MsgInfo_t,INPROC_CONDUIT_CAPACITY_DEFAULT>	*m_dstBuffQueue;
	LockFreeQueue<MsgInfo_t, INPROC_CONDUIT_CAPACITY_DEFAULT> 	*m_dstInnerMsgQueue;
	std::map<int, MsgInfo_t*>	m_dstBuffMap; 
	//
	/*std::vector<boost::latch*>	m_dstOpThreadLatch;*/
	int 	*m_dstOpTokenFlag;
	/*std::atomic<int> 	*m_dstOpThreadAtomic;*/
	//
	std::atomic<int> 	*m_dstUsingPtrFinishFlag;


#ifdef ENABLE_OPBY_FINISH
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
#endif

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
	typedef struct AsyncWorkArgs AsyncWorkArgs_t;
	std::map<int, std::promise<void>> m_srcAsyncReadFinishSet;
	std::map<int, std::promise<void>> m_srcAsyncWriteFinishSet;
	int		*m_srcAsyncOpTokenFlag;
	/*std::atomic<int>	*m_srcAsyncOpThreadAtomic;*/
	int		*m_dstAsyncOpTokenFlag;
	/*std::atomic<int>	*m_dstAsyncOpThreadAtomic;*/
	std::vector<std::atomic<int>*> m_srcAsyncOpThreadAvailable;
	std::vector<std::atomic<int>*> m_srcAsyncOpThreadFinish;
	std::vector<std::atomic<int>*> m_dstAsyncOpThreadAvailable;
	std::vector<std::atomic<int>*> m_dstAsyncOpThreadFinish;

	LockFreeQueue<AsyncWorkArgs_t, INPROC_CONDUIT_CAPACITY_DEFAULT> *m_srcAsyncWorkQueue;
	LockFreeQueue<AsyncWorkArgs_t, INPROC_CONDUIT_CAPACITY_DEFAULT> *m_dstAsyncWorkQueue;
	std::map<int, std::promise<void>> m_dstAsyncReadFinishSet;
	std::map<int, std::promise<void>> m_dstAsyncWriteFinishSet;
	std::atomic<bool> m_srcAsyncWorkerOn;
	std::atomic<bool> m_dstAsyncWorkerOn;

	int m_closeWorkerCountMax = 10000;
#ifdef USE_DEBUG_LOG
	long m_asyncWorkerCount=0;
#endif


    // output debug log to specific file
    static thread_local std::ofstream *m_threadOstream;


	//
	InprocConduit& operator=(const InprocConduit &other)=delete;
    InprocConduit(const InprocConduit &other)=delete;

};



}// end namespace iUtc






#endif


