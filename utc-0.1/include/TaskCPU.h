#ifndef UTC_TASK_CPU_H_
#define UTC_TASK_CPU_H_

#include "UtcBasics.h"
#include "TaskBase.h"
#include "TaskArtifact.h"
#include "TaskInfo.h"
#include "UserTaskBase.h"
#include "FastMutex.h"
#include "FastCond.h"
#include "FastBarrier.h"

#include <vector>
#include <map>
#include "boost/filesystem.hpp"
#include "boost/thread/tss.hpp"
#include "boost/thread/barrier.hpp"
#include "boost/thread/latch.hpp"

namespace iUtc{

class TaskCPU : public TaskArtifact{
public:
	TaskCPU(TaskType taskType,
			int numLocalThreads,
				 int currentProcessRank,
				 int numProcesses,
				 int numTotalThreads,
				 std::vector<ThreadRank_t> tRankList,
				 std::vector<ThreadId_t> *LocalThreadList,
				 std::map<ThreadId_t, ThreadRank_t> *LocalThreadRegistry,
				 std::map<ThreadRank_t, int> *ThreadRank2Local,
				 std::ofstream *procOstream,
				 TaskInfo *commonThreadInfo,
				 ThreadPrivateData *commonThreadPrivateData,
				 boost::thread_specific_ptr<ThreadPrivateData>* threadPrivateData,
				 UserTaskBase* realUserTaskObj);

	~TaskCPU();

	void launchThreads();

	int initImpl(std::function<void()> initHandle);

	int runImpl(std::function<void()> runHandle);

	int execImpl(std::function<void()> execHandle);

	int waitImpl();

	int finishImpl();

	void threadImpl(ThreadRank_t trank, ThreadRank_t lrank,
				std::ofstream* output, int hardcoreId = -1);

	void threadSync();
	void threadSync(ThreadRank_t lrank);

	void threadExit(ThreadRank_t trank);

	void threadWait();

	bool hasActiveLocalThread();
	void waitLocalThreadFinish();

private:
	//
	TaskType m_taskType;

	std::vector<std::thread> m_taskThreads;

	int m_numLocalThreads;

	int m_currentProcessRank;

	int m_numProcesses;

	int m_numTotalThreads;

	std::vector<ThreadRank_t> m_tRankList;

	std::vector<ThreadId_t> *m_LocalThreadList;

    std::map<ThreadId_t, ThreadRank_t> *m_LocalThreadRegistry;

    std::map<ThreadRank_t, int> *m_ThreadRank2Local;

    std::ofstream *m_procOstream;

	enum threadJobType{
		job_init = 0,
		job_run,
		job_finish,
		job_wait,
		job_user_defined
	};
	std::vector<threadJobType> m_jobQueue;
	std::mutex m_jobExecMutex;
	std::condition_variable m_jobExecCond;
	int *m_threadJobIdx;
	boost::barrier *m_threadSync;
	boost::latch *m_jobDoneWait;
	
	/* use a funtion handle queue to store every job's related function
	 * handle. the nullhandle is used to fill queue when push 'finish'
	 * 'wait' jobs, as they do not need creat a function handle, we just
	 * need a nullhandle to fill its position.
	 * Use this queue to deal with if several task.job() call goes queickly,
	 * before thread use the handle to exec for an early call, next call may
	 * already change the handle, if we just use one handle var for same type
	 * of job.
	 *
	 */
	std::vector<std::function<void()>> m_jobHandleQueue;
	std::function<void()> m_nullJobHandle;

	std::function<void()> m_InitHandle;
	std::function<void()> m_RunHandle;
	std::function<void()> m_ExecHandle;

	// used for conduit to check all running threads of a task are finish,
	// and can destroy the conduit
	int m_activeLocalThreadCount;
	//std::mutex m_activeLocalThreadMutex;
	//std::condition_variable m_activeLocalThreadCond;
	FastMutex m_activeLocalThreadMutex;
	FastCond m_activeLocalThreadCond;

	TaskInfo *m_commonTaskInfo;
	ThreadPrivateData *m_commonThreadPrivateData;
	boost::thread_specific_ptr<ThreadPrivateData> *m_threadPrivateData;

	// user task template base obj
	UserTaskBase* m_realUserTaskObj;

};


}// end namespace iUtc


#endif
