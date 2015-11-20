#ifndef UTC_XPROC_CONDUIT_H_
#define UTC_XPROC_CONDUIT_H_

#include "ConduitBase.h"
#include "TaskBase.h"

#include <vector>
#include <map>
#include <mutex>
#include <condition_variable>
#include <fstream>

namespace iUtc{

class XprocConduit: public ConduitBase
{
public:
	XprocConduit(TaskBase* srctask, TaskBase* dsttask, int cdtId);


	/*
	 *  Blocking operation
	 */
	int BWrite(void* DataPtr, DataSize_t DataSize, int tag);
	int BWriteBy(ThreadRank thread, void* DataPtr, DataSize_t DataSize, int tag);
	void BWriteBy_Finish(int tag);

	int Write(void* DataPtr, DataSize_t DataSize, int tag);
	int WriteBy(ThreadRank thread, void* DataPtr, DataSize_t DataSize, int tag);
	void WriteBy_Finish(int tag);

	int PWrite(void* DataPtr, DataSize_t DataSize, int tag);
	int PWriteBy(ThreadRank thread, void* DataPtr, DataSize_t DataSize, int tag);
	void PWriteBy_Finish(int tag);


	int Read(void* DataPtr, DataSize_t DataSize, int tag);
	int ReadBy(ThreadRank thread, void* DataPtr, DataSize_t DataSize, int tag);
	void ReadBy_Finish(int tag);


	/*
	 *  Nonblocking operation
	 */
	int AsyncWrite(void* DataPtr, DataSize_t DataSize, int tag);
	void AsyncWrite_Finish(int tag);


	int AsyncRead(void* DataPtr, DataSize_t DataSize, int tag);
	void AsyncRead_Finish(int tag);


	~XprocConduit();

private:
	int m_conduitId;
	TaskBase* m_srcTask;
	TaskBase* m_dstTask;
	TaskId m_srcId;
	TaskId m_dstId;

	int m_numSrcLocalThreads;
	int m_numDstLocalThreads;

	int m_srcMainResideProc;
	int m_dstMainResideProc;


	// this  means how many msg can be send or recv by different threads when not all threads finish
	// that operation
	int m_noFinishedOpCapacity;
	//
	int m_availableNoFinishedOpCount;
	std::condition_variable m_availableNoFinishedCond;
	std::mutex m_OpCheckMutex;
	int *m_OpRotateCounter;
	int *m_OpRotateCounterIdx;
	int *m_OpRotateFinishFlag;
	std::condition_variable m_OpFinishCond;


	///// for OpBy
	std::map<int, int> m_readbyFinishSet;
	std::mutex m_readbyFinishMutex;
	std::condition_variable m_readbyFinishCond;
	std::map<int, int> m_writebyFinishSet;
	std::mutex m_writebyFinishMutex;
	std::condition_variable m_writebyFinishCond;


	///// for Async op
#ifdef USE_MPI_BASE
	std::map<int, MPI_Request*> m_asyncReadFinishSet;
	std::map<int, MPI_Request*> m_asyncWriteFinishSet;
#endif





	////
	static thread_local std::ofstream *m_threadOstream;

};

}// end namespace iUtc



#endif

