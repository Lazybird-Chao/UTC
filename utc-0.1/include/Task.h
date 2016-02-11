#ifndef UTC_TASK_H_
#define UTC_TASK_H_

#include "UtcBasics.h"
#include "UtcContext.h"
#include "TaskManager.h"
#include "TaskInfo.h"
#include "TaskBase.h"
#include "RankList.h"
#include "UtcException.h"
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

namespace iUtc{

template <class T>
class Task: public TaskBase
{
public:
	typedef T FunctionObjectType;

	Task();
	Task(RankList rList);
	Task( std::string name , RankList rList=RankList(1));

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

	//
	void run();

	void waitTillDone();


	//


protected:
	int initImpl();

	int runImpl();



private:
	//
	void threadImpl(ThreadRank_t trank, ThreadRank_t lrank, std::ofstream* output);

	void threadExit(ThreadRank_t trank);

	void CreateTask(const std::string name,
			const RankList rList);

	void LaunchThreads(std::vector<ThreadRank_t> &tRankList);


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
	int m_threadTerminateSignal;
	int m_calledRun;
	int m_calledInit;

	//
	std::mutex m_threadReady2InitMutex;
	std::condition_variable m_threadReady2InitCond;
	int m_threadReady2InitCounter;

	/*std::mutex m_threadFinishInitMutex;
	std::condition_variable m_threadFinishInitCond;
	int m_threadFinishInitConuter;*/

	std::mutex m_threadSyncInitMutex;
	std::condition_variable m_threadSyncInitCond;
	int m_threadSyncInitCounterComing;
	int m_threadSyncInitCounterLeaving; //a pair of counter used for synch among threads

	//
	std::mutex m_threadReady2RunMutex;
	std::condition_variable m_threadReady2RunCond;
	int m_threadReady2RunCounter;


	/*std::mutex m_threadFinishRunMutex;
	std::condition_variable m_threadFinishRunCond;
	int m_threadFinishRunCounter;*/


	//
	std::function<void()> m_userTaskInitFunctionHandle;

	std::function<void()> m_userTaskRunFunctionHandle;

	//
	Task& operator=(const Task& other)=delete;
	Task(const Task& other)=delete;

};



} //namespace iUtc




// C++ template can not use separate compilation
// template declare and define(implement) shout both in header file
#include "Task.inc"





#endif




