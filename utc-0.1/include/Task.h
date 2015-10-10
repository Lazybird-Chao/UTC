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
#include <functional>
#include <vector>
#include <map>
#include <iostream>
#include <fstream>
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

	//
	void run();

	void waitTillDone();


	//


protected:
	int initImpl();

	int runImpl();



private:
	Task(const Task&);
	Task& operator=(const Task&);

	void threadImpl(ThreadRank trank, std::ofstream* output);

	void threadExit(ThreadRank trank);

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

	/*std::mutex m_threadFinishInitMutex;
	std::condition_variable m_threadFinishInitCond;
	int m_threadFinishInitConuter;*/

	std::mutex m_threadSyncInitMutex;
	std::condition_variable m_threadSyncInitCond;
	int m_threadSyncInitCounter;
	int m_threadSyncInitCounterBar; //a pair of counter used for synch among threads

	//
	std::mutex m_threadReady2RunMutex;
	std::condition_variable m_threadReady2RunCond;
	int m_threadReady2RunCounter;


	/*std::mutex m_threadFinishRunMutex;
	std::condition_variable m_threadFinishRunCond;
	int m_threadFinishRunCounter;*/


	//
	std::function<void()> m_userTaskInitFunctionHandle;



};


// some utility functions
std::ofstream* getProcOstream();

std::ofstream* getThreadOstream();

int getTid();
int getTrank();
int getPrank();
int getLsize();
int getGsize();

} //namespace iUtc




// C++ template can not use separate compilation
// template declare and define(implement) shout both in header file
#include "Task.inc"





#endif




