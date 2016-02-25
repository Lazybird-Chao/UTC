/*
 * test_inprocess_conduit_async.cc
 *
 *  Created on: Feb 24, 2016
 *      Author: chao
 */
#include "Utc.h"
#include <iostream>
#include <string>

using namespace iUtc;


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
        *output<<"taskA do pingpong test..."<<std::endl;
        for(int i=1; i<=SIZE; i=i*4)
        {
            *output<<"\tmessage size: "<<i*sizeof(float)<<" Bytes..."<<std::endl;
            if(!mytrank)
                std::cout<<"\tmessage size: "<<i*sizeof(float)<<" Bytes..."<<std::endl;


            cdt_ptr->AsyncWrite(message_out, i*sizeof(float), i);

            cdt_ptr->AsyncRead(message_in, i*sizeof(float), i);
            if(!mytrank)
            	std::cout<<time_from_start()<<std::endl;
            usleep_for(i*sizeof(float));
            if(!mytrank)
            	std::cout<<time_from_start()<<std::endl;

        }
        *output<<"taskA before wait async..."<<std::endl;
        for(int i=1; i<=SIZE; i=i*4)
        {
        	cdt_ptr->AsyncWrite_Finish(i);
        	cdt_ptr->AsyncRead_Finish(i);
        }
        *output<<"taskA after wait async..."<<std::endl;

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
        *output<<"taskB do pingpong test..."<<std::endl;
        for(int i=1; i<=SIZE; i=i*4)
        {
            *output<<"\tmessage size: "<<i*sizeof(float)<<" Bytes..."<<std::endl;
            cdt_ptr->Read(message_in, i*sizeof(float), i);

            cdt_ptr->Write(message_out, i*sizeof(float), i);
        }
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

    ProcList r_list1(5,0); //1 thread on proc 0
    ProcList r_list2(3,0); //1 thread on proc 1
    Task<user_taskA> task1("ping", r_list1);
    Task<user_taskB> task2("pong", r_list2);
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




