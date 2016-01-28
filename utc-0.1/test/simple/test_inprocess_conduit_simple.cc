#include "Utc.h"
#include <iostream>
#include <string>


using namespace iUtc;

class user_taskA{
public:


	/*void init()
	{
		std::ofstream* output = getThreadOstream();
		*output<<"hello user taskA!"<<std::endl;
		*output<<"doing init taskA..."<<std::endl;
	}*/
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

	}

	void run()
	{
		int mytrank = getTrank();
		int myprank = getPrank();
		std::ofstream* output = getThreadOstream();
		*output<<"doing run taskA..."<<std::endl;
		if(!mytrank)
			*output<<"conduit capacity: "<<cdt_ptr->getCapacity()<<std::endl;

		cdt_ptr->Write((void*)message, 100, 1);
		cdt_ptr->Read(message2, 100,2);
		if(!mytrank)
		    *output<<"Received back: "<<message2<<std::endl;
	}

	Conduit *cdt_ptr;
	char message[100] = "This is the first message I send to you!";
	char message2[100];

};


class user_taskB{
public:


    void init(Conduit *cdt)
    {
    	std::ofstream* output = getThreadOstream();
    	*output<<"hello user taskB!"<<std::endl;
        *output<<"doing init taskB..."<<std::endl;
        if(!getTrank())
        {
        	cdt_ptr = cdt;
        }
    }

    void run()
    {
    	std::ofstream* output = getThreadOstream();
        *output<<"doing run taskB..."<<std::endl;
        Timer timer(MILLISEC);
        timer.start();
        cdt_ptr->Read(message, 100, 1);
        *output<<"Received message: "<<message<<std::endl;
        cdt_ptr->Write(message2, 100, 2);
        *output<<"time cost:"<<timer.stop()<<" ms"<<std::endl;
    }

    Conduit *cdt_ptr;
    char message[100];
    char message2[100]= "Hi, I got that!";
};


int main(int argc, char*argv[])
{

    UtcContext  ctx;

    std::string pname;
    ctx.getProcessorName(pname);
    std::ofstream* pout= getProcOstream();
    *pout<<"proc rank:"<<ctx.getProcRank()<<" processor name:"<<pname.c_str()<<std::endl;

    int t_list[2]={0,0};
    RankList r_list(2, t_list);
    Task<user_taskA> task1("First-task", r_list);
    int t_list2[2]={0,0};
    RankList r_list2(2,t_list2);
    Task<user_taskB> task2("Second-task", r_list2);

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





