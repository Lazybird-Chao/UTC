#include "Utc.h"
#include <iostream>
#include <string>


using namespace iUtc;

class user_task{
public:
	user_task()
	{
		std::cout<<"hello user task!"<<std::endl;
	}

	void init()
	{
		std::cout<<"doing init..."<<std::endl;
	}

	void run()
	{
		std::cout<<"doing run..."<<std::endl;
	}

};


int main(int argc, char*argv[])
{

    UtcContext  ctx;
    //std::cout<<"proc rank:"<<ctx.getProcRank()<<std::endl;
    //std::cout<<"number of procs:"<<ctx.numProcs()<<std::endl;
    std::string pname;
    ctx.getProcessorName(pname);
    std::cout<<"proc rank:"<<ctx.getProcRank()<<" processor name:"<<pname.c_str()<<std::endl;

    int t_list[2]={0,0};
    RankList r_list(2, t_list);
    Task<user_task> task1("First-task", r_list);
    int t_list2[2]={1,1};
    RankList r_list2(2,t_list2);
    Task<user_task> task2("Second-task", r_list2);
    task1.init();

    task1.run();



    return 0;

}





