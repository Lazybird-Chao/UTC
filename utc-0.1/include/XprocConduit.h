#ifndef UTC_XPROC_CONDUIT_H_
#define UTC_XPROC_CONDUIT_H_

#include "ConduitBase.h"
#include "TaskBase.h"

#include <vector>
#include <map>
#include <mutex>
#include <condition_variable>
#include <fstream>
#include <atomic>
#include "boost/thread/latch.hpp"

namespace iUtc{

class XprocConduit: public ConduitBase
{
public:
	XprocConduit(TaskBase* srctask, TaskBase* dsttask, int cdtId, std::string name);


	/*
	 *  Blocking operation
	 */
	int BWrite(void* DataPtr, DataSize_t DataSize, int tag);
	int BWriteBy(ThreadRank_t thread, void* DataPtr, DataSize_t DataSize, int tag);
	void BWriteBy_Finish(int tag){};

	int Write(void* DataPtr, DataSize_t DataSize, int tag);
	int WriteBy(ThreadRank_t thread, void* DataPtr, DataSize_t DataSize, int tag);
	void WriteBy_Finish(int tag){};
	int WriteByFirst(void* DataPtr, DataSize_t DataSize, int tag);


	int PWrite(void* DataPtr, DataSize_t DataSize, int tag);
	int PWriteBy(ThreadRank_t thread, void* DataPtr, DataSize_t DataSize, int tag);
	void PWriteBy_Finish(int tag){};


	int Read(void* DataPtr, DataSize_t DataSize, int tag);
	int ReadBy(ThreadRank_t thread, void* DataPtr, DataSize_t DataSize, int tag);
	void ReadBy_Finish(int tag){};
	int ReadByFirst(void* DataPtr, DataSize_t DataSize, int tag);


	/*
	 *  Nonblocking operation
	 */
	int AsyncWrite(void* DataPtr, DataSize_t DataSize, int tag);
	void AsyncWrite_Finish(int tag);


	int AsyncRead(void* DataPtr, DataSize_t DataSize, int tag);
	void AsyncRead_Finish(int tag);


	~XprocConduit();

private:
	ConduitId_t m_conduitId;
	TaskBase* m_srcTask;
	TaskBase* m_dstTask;
	TaskId_t m_srcId;
	TaskId_t m_dstId;
	std::string m_Name;

	int m_numSrcLocalThreads;
	int m_numDstLocalThreads;

	int m_srcMainResideProc;
	int m_dstMainResideProc;


	/*
	 * using token-ring-net mechanism. Start from thread local rank 0,
	 * each thread responds for doing the r/w one by one
	 */
	/*std::vector<boost::latch*> m_OpThreadLatch;
	std::atomic<int> *m_OpThreadAtomic;*/
	int *m_OpTokenFlag;

	/*
	 * new mechanism for multi-threads op
	 */
	std::vector<std::atomic<int>*> m_OpThreadAvailable;
	std::vector<std::atomic<int>*> m_OpThreadFinish;
	std::vector<boost::latch*> m_OpThreadFinishLatch;
	int m_nOps2=65;

	int *m_OpFirstIdx;
	std::atomic<int> *m_OpThreadIsFirst;
	int m_nOps=33;

#ifdef ENALBE_OPBY_FINISH
	///// for OpByFinish
	std::map<int, int> m_readbyFinishSet;
	std::mutex m_readbyFinishMutex;
	std::condition_variable m_readbyFinishCond;
	std::map<int, int> m_writebyFinishSet;
	std::mutex m_writebyFinishMutex;
	std::condition_variable m_writebyFinishCond;
#endif


	///// for Async op
#ifdef USE_MPI_BASE
	std::map<int, MPI_Request*> m_asyncReadFinishSet;
	std::map<int, MPI_Request*> m_asyncWriteFinishSet;
#endif
	/*std::atomic<int> *m_asyncOpThreadAtomic;*/
	int *m_asyncOpTokenFlag;

	std::vector<std::atomic<int>*> m_asyncOpThreadAvailable;
	std::vector<std::atomic<int>*> m_asyncOpThreadFinish;


	////
	static thread_local std::ofstream *m_threadOstream;


};

}// end namespace iUtc



#endif

