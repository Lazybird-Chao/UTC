/*
 * Basic task creation demo.
 *
 */

/* main UTC header file */
#include "Utc.h"

/* other standard header file */
#include <iostream>
#include <string>
#include <cstdio>
#include <cstring>

/* main UTC namespace */
using namespace iUtc;


std::string prefix;
const int MAX_THREADS = 5;   //max threads of one task in one proc
const int MIN_PROCS = 1;

/*
 * user defined task class
 */
class TaskTest
{
public:
	/*
	 * two member functions that user task must define
	 */
	void init(int testNum);
	void run();

	/*
	 * data members that will be shared by all task-threads
	 */
	int m_testNum;
	std::string m_prefix[MAX_THREADS];
};

void TaskTest::init(int testNum)
{
	/* get current process rank that task stays on */
	int my_proc = getPrank();
	/* get current thread rank within the task */
	int my_thread = getTrank();
	/* get current task object */
	TaskBase* ct = getCurrentTask();
	/* get task name */
	string tname = ct->getName();

	char preost[128];
	std::sprintf(preost, "TaskTest: Test-%d Name:%s : ProcId:[%d] : ThreadRank:[%d]",
			testNum, tname.c_str(), my_proc, my_thread);

	/* For shared data members,usually let one thread initialize them */
	if(getLrank() == 0)
	{
		m_testNum = testNum;
	}
	/* Different thread initialize different part of shared data */
	m_prefix[getLrank()] = preost;

	return;
}

void TaskTest::run()
{
	int my_proc = getPrank();
	int my_thread = getTrank();
	TaskBase* ct = getCurrentTask();
	string tname = ct->getName();

	char preost[128];
	std::sprintf(preost, "TaskTest: Test-%d Name:%s : ProcId:[%d] : ThreadRank:[%d]",
			m_testNum, tname.c_str(), my_proc, my_thread);

	if(std::strcmp(preost, m_prefix[getLrank()].c_str()))
	{
		std::cout<<"ERROR: Incorrect runing context!"<<std::endl;
		std::cout<<preost<<std::endl;
		std::cout<<m_prefix<<std::endl;
	}
	sleep_for(my_proc+1);
	return;
}


/*****************************************************
 * main() program
 ****************************************************/
int main(int argc, char**argv)
{
	/* initialize UTC context */
	UtcContext ctx(argc, argv);

	/* get total procs of UTC runtime */
	int nproc = ctx.numProcs();
	/* get current process rank */
	int myProc = ctx.getProcRank();

	// build a Prefix string for using
	char preost[100];
	sprintf(preost, "TaskTest[%d]: ", myProc);
	prefix = preost;

	sleep_for(myProc+1);
	std::cout<<prefix<<"Starting with "<<nproc<<" Procs"<<std::endl;


	// Test - 1
	// Test Root Task
	/***********************************************/
	TaskId_t ctId = getTaskId();
	TaskBase* ct = getCurrentTask();

	if(ctId != ct->getTaskId())
		std::cout<<"ERROR, current task id mismatch!"<<std::endl;
	if(ctId != 0)
		std::cout<<"ERROR, Root task id should be 0!"<<std::endl;

	TaskId_t ptId = getParentTaskId();
	TaskBase* pt = getParentTask();

	if(ptId != pt->getTaskId() || ptId != ct->getParentTaskId())
		std::cout<<"ERROR, parent task id mismatch!"<<std::endl;
	if(ptId !=0)
		std::cout<<"ERROR, Root's parent task id should be 0!"<<std::endl;

	std::cout<<"Information for RootTask:"<<std::endl;
	ct->display();

	inter_Barrier();
	sleep_for(1);
	std::cout<<std::endl;
	inter_Barrier();




	// Test - 2
	//  One task, and one thread per process
	/********************************************/
	if(nproc > MIN_PROCS)
	{
		/* construct a task map rank list */
		std::vector<Rank_t> rv;
		for(int i=0; i<nproc; i++)
		{
			rv.push_back(i);
		}
		ProcList rl(rv);

		/* define task */
		Task<TaskTest> taskA("TaskA", rl);
		taskA.display();

		/* call init() method of task */
		taskA.init(2);  // this is Test-2
		/* call run() method of task */
		taskA.run();
		std::cout<<prefix<<"Test2 launched TaskA waiting for completion"<<std::endl;
		/* wait for task run finish */
		taskA.waitTillDone();
		std::cout<<prefix<<"Test2 TaskA is done"<<std::endl;
	}
	inter_Barrier();
	sleep_for(1);
	std::cout<<std::endl;
	inter_Barrier();



	// Test - 3
	//  One task, and multiple threads per process
	/********************************************/
	if(nproc > MIN_PROCS)
	{
		std::vector<Rank_t> rv;
		for(int i=0; i<nproc; i++)
		{
			rv.push_back(i);
			rv.push_back(i);
			rv.push_back(i);
		}
		ProcList rl(rv);

		Task<TaskTest> taskA("TaskA", rl);
		taskA.display();

		taskA.init(3); //this is Test-3
		taskA.run();
		std::cout<<prefix<<"Test3 launched TaskA waiting for completion"<<std::endl;
		taskA.waitTillDone();
		std::cout<<prefix<<"Test3 TaskA is done"<<std::endl;
	}
	inter_Barrier();
	sleep_for(1);
	std::cout<<std::endl;
	inter_Barrier();




	// Test - 4
	//  Multiple tasks, and multiple threads per process of each task
	/****************************************************************/
	if(nproc > MIN_PROCS)
	{
		std::vector<Rank_t> rvA;
		std::vector<Rank_t> rvB;
		for (int i=0; i<nproc; i++)
		{
			if ((i%2) == 0)
			{	//   for TaskA
				rvA.push_back(i);
				rvA.push_back(i);
				rvA.push_back(i);
				rvA.push_back(i);
			}
			else
			{	//  for TaskB
				rvB.push_back(i);
				rvB.push_back(i);
				rvB.push_back(i);
			}
		}
		ProcList rlA(rvA);
		ProcList rlB(rvB);
		Task<TaskTest> taskA("TaskA", rlA);
		Task<TaskTest> taskB("TaskB", rlB);
		taskA.display();
		taskB.display();

		taskA.init(4);  //this is Test-4
		taskB.init(4);
		taskA.run();
		taskB.run();
		std::cout<<prefix<<"Test4 launched TaskA & TaskB waiting for completion"<<std::endl;

		taskA.waitTillDone();
		std::cout<<prefix<<"Test4 TaskA is done"<<std::endl;
		taskB.waitTillDone();
		std::cout<<prefix<<"Test4 TaskB is done"<<std::endl;

	}

	return 0;
}
