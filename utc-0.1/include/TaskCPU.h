#ifndef UTC_TASK_CPU_H_
#define UTC_TASK_CPU_H_

#include "UtcBasics.h"
#include "TaskArtifact.h"

#include <vector>

namespace iUtc{

class TaskCPU : public TaskArtifact{
public:
	TaskCPU(int numLocalThreads,
				 int currentProcessRank,
				 std::vector<ThreadRank_t> tRankList,
				 std::vector<ThreadId_t> *LocalThreadList,
				 std::map<ThreadId_t, ThreadRank_t> *LocalThreadRegistry,
				 std::map<ThreadRank_t, int> *ThreadRank2Local,
				 std::ofstream *procOstream);

	int launchThreads();

	int initImpl();

	int runImpl();

	int execImpl();

	int waitImpl();

	int finishImpl();

private:
	//
	std::vector<std::thread> m_taskThreads;

	int m_numLocalThreads;

	int m_currentProcessRank;

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
	

};


}// end namespace iUtc


#endif
