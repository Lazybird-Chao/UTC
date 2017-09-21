/*
 * This is a template base class of User task.
 * You can inherit this class when defining your
 * own task class, and realize the necessary member
 * functions and data members.
 */
#ifndef UTC_USER_TASK_BASE_H_
#define UTC_USER_TASK_BASE_H_

#include "UtcBasics.h"
#include "FastBarrier.h"
#include "FastMutex.h"
#include "SpinLock.h"
#if ENABLE_SCOPED_DATA
	#include "PrivateScopedDataBase.h"
#endif

#if ENABLE_GPU_TASK
	#include "UtcGpuContext.h"
	#include <cuda_runtime.h>
#endif

#include <vector>
#include <map>
#include <mutex>


class UserTaskBase
{
public:

	UserTaskBase();

	virtual ~UserTaskBase();


	/* necessary member functions */
	virtual void initImpl();

	virtual void runImpl();


#if ENABLE_SCOPED_DATA
	/* other useful member functions */
	void registerPrivateScopedData(iUtc::PrivateScopedDataBase* psData);
#endif

	void preInit(int lrank, int trank, int prank,
			int numLocalThreads,
			int numProcesses,
			int numTotalProcesses,
			int numTotalThreads,
			std::map<int,int> *worldRankTranslate,
			std::map<int,int> *groupRankTranslate,
			void *gpuCtx);
	void preExit();

	/* useful data members */
	static thread_local int __localThreadId;
	static thread_local int __globalThreadId;
	static thread_local int __processIdInWorld;
	static thread_local int __processIdInGroup;
	int __numLocalThreads=0;
	int __numGlobalThreads=0;
	int __numWorldProcesses=0;
	int __numGroupProcesses=0;
	std::map<int,int> *__worldRankTranslate=nullptr;
	std::map<int,int> *__groupRankTranslate=nullptr;

	FastBarrier __fastIntraSync;
	FastMutex __fastLock;
	iUtc::SpinLock __fastSpinLock;

#if ENABLE_GPU_TASK
	static thread_local cudaStream_t __streamId;
	static thread_local int __deviceId;

#endif

private:
	/* other useful member functions */


	/* other useful data members */
#if ENABLE_SCOPED_DATA
	std::vector<iUtc::PrivateScopedDataBase *> __psDataRegistry;
#endif
	//std::mutex __opLock;
	FastMutex __opLock;
	iUtc::SpinLock __opSpinLock;



protected:
	/* other useful member functions */


	/* other useful data members */


};
//static thread_local int __localThreadId;
//static thread_local int __globalThreadId;

#endif
