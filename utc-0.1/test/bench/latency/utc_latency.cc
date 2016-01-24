/*
 * utc_latency.cc
 *
 *  Basic latency test of point to point
 *  communication.
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

#define MESSAGE_ALIGNMENT 64
#define MAX_MSG_SIZE (1<<27)
#define MYBUFSIZE (MAX_MSG_SIZE + MESSAGE_ALIGNMENT)
#define SKIP_LARGE  10
#define LOOP_LARGE  100
#define SKIP_LLARGE 1
#define LOOP_LLARGE 10
#define LARGE_MESSAGE_SIZE  8192
#define LLARGE_MESSAGE_SIZE (8192*1024)


/*
 * user defined task class
 */
class SendRecvWorker
{
public:
	/*
	 * two member functions that user task must implement
	 */
	void init(int workerType, Conduit* cdt);

	void run();

	/*
	 * data members
	 */
	char* s_buf_tmp;
	char* r_buf_tmp;
	char* s_buf;
	char* r_buf;
	int worker_type;
	Conduit *m_cdt;
	int skip = 1000;
	int loop = 10000;
};

void SendRecvWorker::init(int workerType, Conduit* cdt)
{
	m_cdt = cdt;
	worker_type = workerType;
	s_buf_tmp = (char*)malloc(MYBUFSIZE);
	s_buf = (char*)(((unsigned long)s_buf_tmp + (MESSAGE_ALIGNMENT-1)) / MESSAGE_ALIGNMENT*MESSAGE_ALIGNMENT);
	r_buf_tmp = (char*)malloc(MYBUFSIZE);
	r_buf = (char*)(((unsigned long)r_buf_tmp + (MESSAGE_ALIGNMENT-1)) / MESSAGE_ALIGNMENT*MESSAGE_ALIGNMENT);
	if(workerType == 0)
	{
		// this is the sender

		for(int i =0; i<MAX_MSG_SIZE; i++)
		{
			s_buf[i]='s';
		}
	}
	else if(workerType == 1)
	{
		// this is the receiver

		for(int i =0; i<MAX_MSG_SIZE; i++)
		{
			s_buf[i]='r';
		}
	}
	else
	{
		std::cerr<<"undefined worker type!"<<std::endl;
	}

	return;
}

void SendRecvWorker::run()
{
	int size;
	Timer timer;
	double cost_time;
	//char end_data;

	if(worker_type == 0)
	{
		std::cout<<"Byte"<<"\t\t"<<"us"<<std::endl;
		for(size = 0; size <= MAX_MSG_SIZE; size = (size?size*2:1))
		{
			if(size > LARGE_MESSAGE_SIZE)
			{
				loop = LOOP_LARGE;
				skip = SKIP_LARGE;
			}
			else if(size > LLARGE_MESSAGE_SIZE)
			{
				loop = LOOP_LLARGE;
				skip = SKIP_LLARGE;
			}
			int i=0;
			for(i=0; i<loop+skip; i++)
			{
				if(i==skip)
					timer.start();
				m_cdt->PWrite(s_buf, size, i);
				m_cdt->Read(r_buf, size, i);
			}
			//m_cdt->Read(&end_data, 1 , i);
			cost_time = timer.stop();

			double latency = cost_time*1e6/(2*loop);
			std::cout<<size<<"\t\t"<<latency<<std::endl;

		}
	}
	else if(worker_type == 1)
	{
		//end_data = 'e';
		for(size =0; size<=MAX_MSG_SIZE; size=(size?size*2:1))
		{
			if(size > LARGE_MESSAGE_SIZE)
			{
				loop = LOOP_LARGE;
				skip = SKIP_LARGE;
			}
			else if(size > LLARGE_MESSAGE_SIZE)
			{
				loop = LOOP_LLARGE;
				skip = SKIP_LLARGE;
			}
			int i;
			for(i=0; i<loop+skip; i++)
			{
				m_cdt->Read(r_buf, size, i);
				m_cdt->PWrite(s_buf, size, i);
			}
			//m_cdt->Write(&end_data, 1, i);
		}
	}

	return;
}


/*****************************************************
 * main() program
 ****************************************************/
int main(int argc, char* argv[])
{
	/* initialize UTC context */
	UtcContext ctx(argc, argv);

	/* get total procs of UTC runtime */
	int nproc = ctx.numProcs();
	/* get current process rank */
	int myProc = ctx.getProcRank();

	/* define sender and receiver task obj */
	RankList rl1(2, 0);
	Task<SendRecvWorker> sender(rl1);
	RankList rl2(1, 0);
	Task<SendRecvWorker> receiver(rl2);

	/* define conduit obj */
	Conduit sr_cdt(&sender, &receiver);

	sender.init(0, &sr_cdt);
	receiver.init(1, &sr_cdt);
	sender.run();
	receiver.run();
	sender.waitTillDone();
	receiver.waitTillDone();

	return 0;
}





