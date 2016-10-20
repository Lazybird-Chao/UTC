/*
 * multi-task single thread matrix multiply
 */

/*
 * single task multi-thread matrix multiply
 *
 */
#include "Utc.h"
#include <cstdlib>

using namespace iUtc;


#define size_N  1024
#define subsize_N (1024/2)

class user_task_subMM{
public:
    void init(Conduit* cdt);

    void run();

    float *submatrix_A = new float[subsize_N*size_N];
    float *submatrix_B = new float[size_N*subsize_N];
    float *submatrix_C = new float[subsize_N*size_N];

    ~user_task_subMM()
    {
        delete submatrix_A;
        delete submatrix_B;
        delete submatrix_C;
    }
    Conduit *cdt_ptr;

};

void user_task_subMM::init(Conduit* cdt)
{
    std::ofstream* output = getThreadOstream();
    *output<<" start initializing matrices with random numbers."<<std::endl;
    for(int i=0; i< subsize_N;i++)
    {
        for(int j=0; j< size_N; j++)
        {
            submatrix_A[i*size_N + j] = 4;//(float)rand()/size_N;
            submatrix_B[i*size_N + j] = 2;//(float)rand()/size_N;

        }
    }

    *output<<" finish initialize."<<std::endl;
    if(!getTrank())
    {
        cdt_ptr = cdt;

    }

}
void user_task_subMM::run()
{
    std::ofstream* output = getThreadOstream();
    *output<<" start computing: "<<std::endl;
    Timer timer;
    timer.start();
    *output<<"\t computing with local data..."<<std::endl;
    for(int i =0; i< subsize_N; i++)
    {
        for(int j= 0; j<subsize_N; j++)
        {
            float tmp = 0.0;
            for(int k =0; k<size_N; k++)
            {
                tmp += submatrix_A[i*size_N + k]*submatrix_B[j*size_N +k];
            }
            submatrix_C[i*size_N + j]=tmp;
        }

    }
    *output<<"\t sending local data to another task"<<std::endl;
    cdt_ptr->Write((void*)submatrix_B, subsize_N*size_N*sizeof(float), 123);
    *output<<"\t getting another part of data..."<<std::endl;
    float *B = new float[subsize_N*size_N];
    cdt_ptr->Read((void*)B, subsize_N*size_N*sizeof(float), 123);
    *output<<"\t computing with remote data..."<<std::endl;
    for(int i =0; i< subsize_N; i++)
    {
       for(int j= 0; j<subsize_N; j++)
       {
           float tmp = 0.0;
           for(int k =0; k<size_N; k++)
           {
               tmp += submatrix_A[i*size_N + k]*B[j*size_N +k];
           }
           /*if(tmp != submatrix_C[i*size_N + j])
               *output<<"error on "<<i<<" "<< j<<std::endl;*/
           submatrix_C[i*size_N + j + subsize_N]=tmp;
       }

    }
    delete B;
    double t1 = timer.stop();

    *output<<"time cost: "<<t1<<" s"<<std::endl;
}

int main()
{
    UtcContext ctx;
    std::string pname;
    ctx.getProcessorName(pname);
   std::ofstream* pout= getProcOstream();
   *pout<<"proc rank:"<<ctx.getProcRank()<<" processor name:"<<pname.c_str()<<std::endl;

    RankList r_list(1);
    Task<user_task_subMM> task1("para-MM1", r_list);
    Task<user_task_subMM> task2("para-MM2", r_list);
    Conduit cdt(&task1, &task2);
    Timer timer;
    task1.init(&cdt);
    task2.init(&cdt);

    timer.start();
    task1.run();
    task2.run();
    double t1=timer.stop();

    task1.waitTillDone();
    task2.waitTillDone();
    double t2= timer.stop();

    *pout<<"time1: "<< t1<<" s"<<std::endl;
    *pout<<"time2: "<< t2<<" s"<<std::endl;

    return 0;
}



