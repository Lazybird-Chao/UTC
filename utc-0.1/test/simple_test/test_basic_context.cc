#include "Utc.h"
#include <iostream>
#include <string>
#include <thread>

using namespace iUtc;

int main(int argc, char*argv[])
{

    UtcContext  cont;
    std::cout<<"proc rank:"<<cont.getProcRank()<<std::endl;
    std::cout<<"proc number:"<<cont.numProcs()<<std::endl;
    std::string pname;
    cont.getProcessorName(pname);
    std::cout<<"processor name:"<<pname.c_str()<<std::endl;

    TaskManager* tsm = cont.getTaskManager();
    std::vector<ThreadId> tlist =  tsm->getCurrentTask()->getLocalThreadList();
    for(auto& itor:tlist)
        std::cout<<itor<<" ";
    std::cout<<std::endl;
    std::cout<< TaskManager::getThreadId()<<std::endl;
    std::cout<< std::this_thread::get_id()<<std::endl;


    return 0;

}
