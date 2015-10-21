#include "Utc.h"
#include <iostream>
#include <string>
#include <unistd.h>

using namespace iUtc;

class user_task1Barrier{
public:

	void init()
	{
		std::ofstream* output = getThreadOstream();
		*output<<"this is barrier task!"<<std::endl;

	}

	void run()
	{
		int mytrank = getTrank();
		int myprank = getPrank();
		std::ofstream* output = getThreadOstream();

		sleep(mytrank+1);
		std::cout<<"task1 thread "<<mytrank<<" slept for "<<mytrank+1<<" seconds."<<std::endl;

		intra_Barrier();

		std::cout<<"task1 thread "<<mytrank<<" finish."<<std::endl;

	}
};
class user_task2Barrier{
public:

	void init()
	{
		std::ofstream* output = getThreadOstream();
		*output<<"this is barrier task!"<<std::endl;

	}

	void run()
	{
		int mytrank = getTrank();
		int myprank = getPrank();
		std::ofstream* output = getThreadOstream();

		sleep(mytrank+6);
		std::cout<<"task2 thread "<<mytrank<<" slept for "<<mytrank+6<<" seconds."<<std::endl;

		intra_Barrier();

		std::cout<<"task2 thread "<<mytrank<<" finish."<<std::endl;

	}
};


int main()
{
	UtcContext ctx;

	std::string pname;
	ctx.getProcessorName(pname);
	std::ofstream* pout= getProcOstream();
	*pout<<"proc rank:"<<ctx.getProcRank()<<" processor name:"<<pname.c_str()<<std::endl;

	int t_list[5]={0,0,0,0,0};
	RankList r_list(5, t_list);
	Task<user_task1Barrier> task1("btask1", r_list);
	RankList r_list2(3);
	Task<user_task2Barrier> task2("btask2", r_list2);

	task1.init();
	task2.init();

	task1.run();
	task2.run();

	return 0;
}

