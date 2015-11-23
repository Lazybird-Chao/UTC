/*
 * Basic conduit creation and using demo.
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


const int MAX_THREADS = 5;

/*
 * user defined task class
 */
class RWtask: public UserTaskBase
{
public:
	/*
	 * two member functions that user task must implement
	 */
	void init(Conduit* cdt, int testNum, int sleep_time);
	void run();

	/*
	 * data members that will be shared by all task-threads
	 */
	int m_testNum;
	int m_time;
	char greeting_msg[MAX_THREADS][100];
	char recv_buff[100];
	Conduit* m_cdt;

};

void RWtask::init(Conduit* cdt, int testNum, int sleep_time)
{
	/* get current process rank that task stays on */
	int my_proc = getPrank();
	/* get current thread rank within the task */
	int my_thread = getTrank();
	/* get current thread local rank in the proc */
	int local_rank = getLrank();
	/* get current task object */
	TaskBase* ct = getCurrentTask();
	/* get task name */
	string tname = ct->getName();

	/* initialize data members */
	// different thread can access different part of shared data member at same time.
	std::sprintf(greeting_msg[local_rank],
			"Test-%d, greeting message from:<Name:%s : ProcId:[%d] : ThreadRank:[%d]>",
			testNum, tname.c_str(), my_proc, my_thread);
	// only one thread access the whole shared data member at one time.
	if(local_rank ==0)
	{
		m_testNum = testNum;
		m_time = sleep_time;
		m_cdt = cdt;
	}
	return;
}

void RWtask::run()
{
	int my_proc = getPrank();
	int my_thread = getTrank();
	int local_rank = getLrank();
	TaskBase* ct = getCurrentTask();
	string tname = ct->getName();

	/* write message to conduit */
	m_cdt->Write(greeting_msg[local_rank], 100, 1);
	sleep_for(m_time);
	/* read message from conduit */
	m_cdt->Read(recv_buff, 100, 1);
	std::cout<<tname<<"got message:\n"<<"\t\t"<<recv_buff<<"\n"<<std::endl;

	return;
}

/*****************************************************
 * main() program
 ****************************************************/
int main()
{
	/* initialize UTC context */
	UtcContext ctx();
	/* get total procs of UTC runtime */
	int nproc = ctx.numProcs();
	if(nproc < 2)
	{
		std::cout<<"ERROR, this test needs 2 processes!"<<std::endl;
		return 1;
	}

	// Test - 1
	// Conduit between two tasks in one process
	/***********************************************/



	// Test - 1
	// Conduit between two tasks in two process
	/***********************************************/


	return 0;
}
