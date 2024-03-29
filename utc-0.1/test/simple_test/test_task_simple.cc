#include "Utc.h"
#include <iostream>
#include <fstream>
#include <string>


using namespace iUtc;

class user_taskA{
public:
	user_taskA()
	{

	}

	void init()
	{
		std::ofstream* output = getThreadOstream();
		*output<<"hello user taskA!"<<std::endl;
		*output<<"doing init taskA..."<<std::endl;
	}

	void run()
	{
		std::ofstream* output = getThreadOstream();
		*output<<"doing run taskA..."<<std::endl;
	}

};


class user_taskB{
public:
    user_taskB()
    {

    }

    void init()
    {
    	std::ofstream* output = getThreadOstream();
    	*output<<"hello user taskB!"<<std::endl;
        *output<<"doing init taskB..."<<std::endl;
    }

    void run()
    {
    	std::ofstream* output = getThreadOstream();
        *output<<"doing run taskB..."<<std::endl;
    }

};


int main(int argc, char*argv[])
{

    UtcContext  ctx;
    //std::cout<<"proc rank:"<<ctx.getProcRank()<<std::endl;
    //std::cout<<"number of procs:"<<ctx.numProcs()<<std::endl;
    std::string pname;
    ctx.getProcessorName(pname);
    std::ofstream* pout= getProcOstream();
    *pout<<"proc rank:"<<ctx.getProcRank()<<" processor name:"<<pname.c_str()<<std::endl;

    int t_list[2]={0,0};
    RankList r_list(2, t_list);
    Task<user_taskA> task1("First-task", r_list);
    int t_list2[4]={1, 1, 2, 2};
    RankList r_list2(4,t_list2);
    Task<user_taskB> task2("Second-task", r_list2);


    task1.init();

    task1.run();

    task2.init();

    task2.run();

    task1.waitTillDone();


    return 0;

}





