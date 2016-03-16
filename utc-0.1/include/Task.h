#ifndef UTC_TASK_H_
#define UTC_TASK_H_

#include "UtcBasics.h"
#include "UtcContext.h"
#include "TaskManager.h"
#include "TaskInfo.h"
#include "TaskBase.h"
#include "UtcException.h"
#include "ProcList.h"
#include "Barrier.h"

#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cassert>
#include <sys/types.h>
#include <unistd.h>
#include "boost/filesystem.hpp"
#include "boost/thread/tss.hpp"
#include "boost/thread/barrier.hpp"
#include "boost/thread/latch.hpp"


namespace iUtc{

template <class T>
class Task: public TaskBase
{
public:
	typedef T FunctionObjectType;

	Task();
	Task(ProcList rList);
	Task(std::string name);
	Task( std::string name , ProcList rList);

	~Task();


	//
	void init();

	template<typename T1>
	void init(T1 arg1);

	template<typename T1, typename T2>
	void init(T1 arg1, T2 arg2);

	template<typename T1, typename T2, typename T3>
	void init(T1 arg1, T2 arg2, T3 arg3);

	template<typename T1, typename T2, typename T3, typename T4>
	void init(T1 arg1, T2 arg2, T3 arg3, T4 arg4);

	template<typename T1, typename T2, typename T3, typename T4, typename T5>
	void init(T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5);

	template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
	void init(T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6);

	template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7>
	void init(T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7);

	template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8>
	void init(T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8);

	template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9>
	void init(T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9);

	template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10>
	void init(T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, T10 arg10);
	//
	void run();
	//
	void wait();
	//
	void finish();
	//
	void exec(void (T::*user_fun)());
	template<typename T1>
	void exec(void (T::*user_fun)(T1), T1 arg1);


	//


protected:
	int initImpl();

	int runImpl();

	int execImpl();


private:
	//
	void threadImpl(ThreadRank_t trank, ThreadRank_t lrank, std::ofstream* output);

	void threadExit(ThreadRank_t trank);

	void CreateTask(const std::string name,
			const ProcList rList);

	void LaunchThreads(std::vector<ThreadRank_t> &tRankList);

	void threadSync();
	void threadSync(ThreadRank_t lrank);

	void threadWait();


	//
	std::vector<std::thread> m_taskThreads;

	// barrier obj, shared by all threads in one task
	Barrier *m_taskBarrierObjPtr;

	// mpi comm related obj
#ifdef USE_MPI_BASE
	MPI_Comm m_taskComm;
	MPI_Group m_taskmpiGroup;
#endif


	// user task obj, shared by all threads in one task
	T *m_userTaskObjPtr;

	//
	bool callTaskFinish;

	//
	boost::barrier *m_threadSync;

	//
	std::function<void()> m_userTaskInitFunctionHandle;

	std::function<void()> m_userTaskRunFunctionHandle;

	std::function<void()> m_userTaskExecHandle;
	/* use a funtion handle queue to store every job's related function
	 * handle. the nullhandle is used to fill queue when push 'finish'
	 * 'wait' jobs, as they do not need creat a function handle, we just
	 * need a nullhandle to fill its position.
	 * Sse this queue to deal with if several task.job() call goes queickly,
	 * before thread use the handle to exec for an early call, next call may
	 * already change the handle, if we just use one handle var for same type
	 * of job.
	 *
	 */
	std::vector<std::function<void()>> m_jobHandleQueue;
	std::function<void()> m_nullJobHandle;

	//
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

	boost::latch *m_jobDoneWait;

	//
	Task& operator=(const Task& other)=delete;
	Task(const Task& other)=delete;

};



} //namespace iUtc




// C++ template can not use separate compilation
// template declare and define(implement) shout both in header file
#include "Task.inc"





#endif




