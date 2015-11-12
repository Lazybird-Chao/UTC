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


	int BWrite(void* DataPtr, int DataSize, int tag);
	int BWriteBy(ThreadRank thread, void* DataPtr, int DataSize, int tag);
	void BWriteBy_Finish(int tag);

	int Write(void* DataPtr, int DataSize, int tag);
	int WriteBy(ThreadRank thread, void* DataPtr, int DataSize, int tag);
	void WriteBy_Finish(int tag);

	int PWrite(void* DataPtr, int DataSize, int tag);
	int PWriteBy(ThreadRank thread, void* DataPtr, int DataSize, int tag);
	void PWriteBy_Finish(int tag);


	int Read(void* DataPtr, int DataSize, int tag);
	int ReadBy(ThreadRank thread, void* DataPtr, int DataSize, int tag);
	void ReadBy_Finish(int tag);

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

	// this has different meaning as inproc conduit. There, the capacity is the number of
	// msg that sender write into conduit but hasn't be read by receiver.
	// here it means how many msg can be send or recv when not all threads finish
	// the operation
	int m_capacity;
	//
	int m_availableNoFinishedWriteOpCount;
	std::condition_variable m_availableNoFinishedWriteCond;
	std::mutex m_WriteOpCheckMutex;
	int *m_WriteOpRotateCounter;
	int *m_WriteOpRotateCounterIdx;
	int *m_WriteOpRotateFinishFlag;
	std::condition_variable m_WriteOpFinishCond;

	int m_availableNoFinishedReadOpCount;
	std::condition_variable m_availableNoFinishedReadCond;
	std::mutex m_ReadOpCheckMutex;
	int *m_ReadOpRotateCounter;
	int *m_ReadOpRotateCounterIdx;
	int *m_ReadOpRotateFinishFlag;
	std::condition_variable m_ReadOpFinishCond;



	/////
	std::map<int, int> m_readbyFinishSet;
	std::mutex m_readbyFinishMutex;
	std::condition_variable m_readbyFinishCond;
	std::map<int, int> m_writebyFinishSet;
	std::mutex m_writebyFinishMutex;
	std::condition_variable m_writebyFinishCond;


	static thread_local std::ofstream *m_threadOstream;

};

}// end namespace iUtc



#endif
