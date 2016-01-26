/*
 * utc_bw.cc
 *
 */

/* main UTC header file */
#include "Utc.h"

/* other standard header file */
#include <iostream>
#include <string>
#include <cstdio>
#include <cstring>
#include <unistd.h>

/* main UTC namespace */
using namespace iUtc;

#define MAX_ALIGNMENT 65536
#define MAX_MSG_SIZE (1<<27)
#define MYBUFSIZE (MAX_MSG_SIZE + MAX_ALIGNMENT)

#define LOOP_LARGE 20
#define WINDOW_SIZE_LARGE 64
#define SKIP_LARGE 2
#define LARGE_MESSAGE_SIZE (1<<16)
#define LOOP_LLARGE 10
#define SKIP_LLARGE 1
#define LLARGE_MESSAGE_SIZE (1<<23)
#define WINDOW_SIZE_LLARGE 4
int align_size = getpagesize();

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
	int r_buf_for_sender;
	int s_buf_for_receiver;
	int worker_type;
	Conduit *m_cdt;
	int skip = 10;
	int loop = 100;
	int window_size = 64;
};

void SendRecvWorker::init(int workerType, Conduit* cdt)
{
	m_cdt = cdt;
	worker_type = workerType;
	s_buf_tmp = (char*)malloc(MYBUFSIZE);
	s_buf = (char*)(((unsigned long)s_buf_tmp + (align_size-1)) / align_size*align_size);
	r_buf_tmp = (char*)malloc(MYBUFSIZE);
	r_buf = (char*)(((unsigned long)r_buf_tmp + (align_size-1)) / align_size*align_size);
	if(workerType == 0)
	{
		// this is the sender

		for(int i =0; i<MAX_MSG_SIZE; i++)
		{
			s_buf[i]='s';
		}
		r_buf_for_sender=1;
	}
	else if(workerType == 1)
	{
		// this is the receiver

		for(int i =0; i<MAX_MSG_SIZE; i++)
		{
			s_buf[i]='r';
		}
		s_buf_for_receiver=2;
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

	if(worker_type==0)
	{
		std::cout<<"Byte"<<"\t\t"<<"Mbit/s"<<std::endl;
		for(size=1; size<=MAX_MSG_SIZE;size*=2)
		{
			if(size>LARGE_MESSAGE_SIZE)
			{
				loop = LOOP_LARGE;
				skip = SKIP_LARGE;
				window_size = WINDOW_SIZE_LARGE;
			}
			else if(size>LLARGE_MESSAGE_SIZE)
			{
				loop=LOOP_LLARGE;
				skip=SKIP_LLARGE;
				window_size = WINDOW_SIZE_LLARGE;
			}

			int i,j;
			for(i=0; i<loop+skip; i++)
			{
				if(i==skip)
					timer.start();
				//std::cout<<"here"<<std::endl;
				for(j=0; j<window_size; j++)
					m_cdt->AsyncWrite(s_buf,size,i*window_size+j);
				//std::cout<<"here"<<std::endl;
				for(j=0; j<window_size; j++)
					m_cdt->AsyncWrite_Finish(i*window_size+j);
				m_cdt->Read(&r_buf_for_sender,4,i*window_size+j);

			}
			cost_time=timer.stop();
			double bw = size/1e6*loop*window_size*8;
			bw = bw/cost_time;
			std::cout<<size<<"\t\t"<<bw<<std::endl;
		}
	}
	else if(worker_type==1)
	{
		for(size=1; size<=MAX_MSG_SIZE;size*=2)
		{
			if(size>LARGE_MESSAGE_SIZE)
			{
				loop = LOOP_LARGE;
				skip = SKIP_LARGE;
				window_size = WINDOW_SIZE_LARGE;
			}
			else if(size>LLARGE_MESSAGE_SIZE)
			{
				loop=LOOP_LLARGE;
				skip=SKIP_LLARGE;
				window_size = WINDOW_SIZE_LLARGE;
			}

			int i,j;
			for(i=0; i<loop+skip; i++)
			{
				for(j=0; j<window_size; j++)
					m_cdt->AsyncRead(r_buf,size,i*window_size+j);
				for(j=0; j<window_size; j++)
					m_cdt->AsyncRead_Finish(i*window_size+j);
				m_cdt->Write(&s_buf_for_receiver,4,i*window_size+j);
			}
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
	RankList rl1(1, 0);
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
