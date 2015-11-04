#include "Utc.h"
#include <iostream>
#include <string>


using namespace iUtc;

#define TEST_BWRITE
//#define TEST_WRITE
//#define TEST_PWRITE

#define SIZE (1024*1024)

class user_taskA{
public:
	void init(Conduit * cdtup, Conduit *cdtdown)
	{
		std::ofstream* output = getThreadOstream();
		*output<<"hello user conduit taskA!"<<std::endl;
		*output<<"doing init taskA..."<<std::endl;
		*output<<"conduit: id "<<cdtup->getConduitId()<<", name '"<<cdtup->getName()<<"'"<<std::endl;
		*output<<"conduit: id "<<cdtdown->getConduitId()<<", name '"<<cdtdown->getName()<<"'"<<std::endl;
		if(!getTrank())
		{
			cdtup_ptr = cdtup;
			cdtdown_ptr = cdtdown;
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
		*output<<"task do ring transfer test..."<<std::endl;
		for(int i=1; i<=SIZE;i=4*i)
		{
			*output<<"\tmessage size: "<<i*sizeof(float)<<" Bytes..."<<std::endl;
			if(getTid() ==1 && mytrank==0)
				std::cout<<"\tmessage size: "<<i*sizeof(float)<<" Bytes..."<<std::endl;
#ifdef TEST_BWRITE
			cdtdown_ptr->BWrite(message_out, i*sizeof(float), i);
#elif TEST_WRITE
			cdtdown_ptr->Write(message_out, i*sizeof(float), i);
#elif TEST_PWRITE
			cdtdown_ptr->PWrite(message_out, i*sizeof(float), i);
#endif
			cdtup_ptr->Read(message_in, i*sizeof(float), i);

			//intra_Barrier();
		}
	}

	Conduit *cdtup_ptr;
	Conduit *cdtdown_ptr;
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

	RankList r_list1(5);
	Task<user_taskA> task1("ring1", r_list1);
	Task<user_taskA> task2("ring2", r_list1);
	Task<user_taskA> task3("ring3", r_list1);
	Task<user_taskA> task4("ring4", r_list1);
	Task<user_taskA> task5("ring5", r_list1);
	Conduit cdt12(&task1, &task2);
	Conduit cdt23(&task2, &task3);
	Conduit cdt34(&task3, &task4);
	Conduit cdt45(&task4, &task5);
	Conduit cdt51(&task1, &task5);
	Timer timer(MILLISEC);

	timer.start();
	task1.init(&cdt51, &cdt12);
	task2.init(&cdt12, &cdt23);
	task3.init(&cdt23, &cdt34);
	task4.init(&cdt34, &cdt45);
	task5.init(&cdt45, &cdt51);

	task1.run();
	task2.run();
	task3.run();
	task4.run();
	task5.run();
	double t1 = timer.stop();

	task1.waitTillDone();
	task2.waitTillDone();
	task3.waitTillDone();
	task4.waitTillDone();
	task5.waitTillDone();
	double t2 = timer.stop();

	*pout<<t1<<std::endl<<t2<<std::endl;

	return 0;


}
