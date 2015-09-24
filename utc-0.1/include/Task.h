#ifndef UTC_TASK_H_
#define UTC_TASK_H_

#include "UtcBasics.h"
#include "UtcContext.h"
#include "TaskManager.h"
#include "TaskInfo.h"
#include "TaskBase.h"
#include "RankList.h"
#include "UtcException.h"

#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <map>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <sys/types.h>
#include <unistd.h>

namespace iUtc{

template <class T>
class Task: public TaskBase
{
public:
	typedef T FunctionObjectType;

	Task( std::string name = "Task",
			 RankList rList= RankList(1));

	~Task();


	//
	void init();

	void run();

	void waitTillDone();

protected:
	int initImpl();

	int runImpl();



private:
	Task(const Task&);
	Task& operator=(const Task&);

	void threadImpl(ThreadRank trank);

	void CreateTask(const std::string name,
			const RankList rList);

	void LaunchThreads(std::vector<ThreadRank> &tRankList);

	//
	std::vector<std::thread> m_taskThreads;

	T *m_userTaskObjPtr;

	//
	int m_threadTerminateSignal;
	int m_calledRun;
	int m_calledInit;

	//
	std::mutex m_threadReady2InitMutex;
	std::condition_variable m_threadReady2InitCond;
	int m_threadReady2InitCounter;

	//
	std::mutex m_threadReady2RunMutex;
	std::condition_variable m_threadReady2RunCond;
	int m_threadReady2RunCounter;

	//
	std::mutex m_threadFinishRunMutex;
	std::condition_variable m_threadFinishRunCond;
	int m_threadFinishRunCounter;


};


} //namespace iUtc




// C++ template can not use separate compilation
// template declare and define(implement) shout both in header file
#include "Task.inc"





#endif




