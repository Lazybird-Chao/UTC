#include "Utc.h"
#include <iostream>
#include <string>

using namespace iUtc;

//#define TEST_BWRITE
#define TEST_WRITE
//#define TEST_PWRITE

#define SIZE (1024*1024)

class user_taskA{
public:
	void init(Conduit *cdt)
	{
		std::ofstream* output = getThreadOstream();
		*output<<"hello user conduit taskA!"<<std::endl;
		*output<<"doing init taskA..."<<std::endl;
		*output<<"conduit: id "<<cdt->getConduitId()<<" name '"<<cdt->getName()<<"'"<<std::endl;
		if(!getTrank())
		{
			cdt_ptr = cdt;
		}
		if(getTrank() ==0)
		{
			message_out = new float[SIZE];
			for(int i=0; i<SIZE; i++)
			{
				message_out[i]= i;
			}
			message_in = new float[SIZE];
		}

	}

	void run()
	{
		std::ofstream* output = getThreadOstream();
		int mytrank = getTrank();
		*output<<"taskA do RWby test..."<<std::endl;
		for(int i=1; i<=SIZE; i=i*4)
		{
			*output<<"\tmessage size: "<<i*sizeof(float)<<" Bytes..."<<std::endl;
			if(!mytrank)
				std::cout<<"\tmessage size: "<<i*sizeof(float)<<" Bytes..."<<std::endl;
#if defined(TEST_BWRITE)
			cdt_ptr->BWriteBy(0,message_out, i*sizeof(float), i);
#elif defined(TEST_WRITE)
			cdt_ptr->WriteBy(0,message_out, i*sizeof(float), i);
#elif defined(TEST_PWRITE)
			cdt_ptr->PWriteBy(0,message_out, i*sizeof(float), i);
#endif
			cdt_ptr->ReadBy(2,message_in, i*sizeof(float), i);
			//intra_Barrier();
		}
		cdt_ptr->ReadBy_Finish(SIZE);
#if defined(TEST_BWRITE)
		cdt_ptr->BWriteBy_Finish(SIZE);
#elif defined(TEST_WRITE)
		cdt_ptr->WriteBy_Finish(SIZE);
#elif defined(TEST_PWRITE)
		cdt_ptr->PWriteBy_Finish(SIZE);
#endif
		*output<<"end RWby!"<<std::endl;
	}
	Conduit *cdt_ptr;
	float *message_out;
	float *message_in;
};

class user_taskB{
public:
	void init(Conduit *cdt)
	{
		std::ofstream* output = getThreadOstream();
		*output<<"hello user conduit taskB!"<<std::endl;
		*output<<"doing init taskB..."<<std::endl;
		*output<<"conduit: id "<<cdt->getConduitId()<<" name '"<<cdt->getName()<<"'"<<std::endl;
		if(!getTrank())
		{
			cdt_ptr = cdt;
		}
		if(getTrank() ==0)
		{
			message_out = new float[SIZE];
			for(int i=0; i<SIZE; i++)
			{
				message_out[i]= -1*i;
			}
			message_in = new float[SIZE];
		}

	}

	void run()
	{
		std::ofstream* output = getThreadOstream();
		int mytrank = getTrank();
		*output<<"taskB do RWby test..."<<std::endl;
		for(int i=1; i<=SIZE; i=i*4)
		{
			*output<<"\tmessage size: "<<i*sizeof(float)<<" Bytes..."<<std::endl;
			cdt_ptr->Read(message_in, i*sizeof(float), i);
#if defined(TEST_BWRITE)
			cdt_ptr->BWrite(message_out, i*sizeof(float), i);
#elif defined(TEST_WRITE)
			cdt_ptr->Write(message_out, i*sizeof(float), i);
#elif defined(TEST_PWRITE)
			cdt_ptr->PWrite(message_out, i*sizeof(float), i);
#endif
			//intra_Barrier();
		}
		*output<<"end RWby!"<<std::endl;
	}
	Conduit *cdt_ptr;
	float *message_out;
	float *message_in;
};

int main()
{
	UtcContext ctx;

	std::string pname;
	ctx.getProcessorName(pname);
	std::ofstream* pout= getProcOstream();
	*pout<<"proc rank:"<<ctx.getProcRank()<<" processor name:"<<pname.c_str()<<std::endl;

	ProcList r_list1(3,0);
	Task<user_taskA> task1("writer", r_list1);
	ProcList r_list2(2,0);
	Task<user_taskB> task2("reader", r_list2);
	Conduit cdt1(&task1, &task2);
	Timer timer(MILLISEC);

	timer.start();

	task1.init(&cdt1);

	task1.run();

	task2.init(&cdt1);

	task2.run();
	double t1 = timer.stop();

	task1.waitTillDone();
	task2.waitTillDone();
	double t2 = timer.stop();

	*pout<<t1<<std::endl<<t2<<std::endl;

	return 0;

}

