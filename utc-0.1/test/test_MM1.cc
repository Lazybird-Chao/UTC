/*
 * single task multi-thread matrix multiply
 *
 */
#include "Utc.h"
#include <cstdlib>

using namespace iUtc;


#define size_N  1024

class user_taskMM{
public:
    void init();

    void run();

    float *matrix_A = new float[size_N*size_N];
    float *matrix_B = new float[size_N*size_N];
    float *matrix_C = new float[size_N*size_N];

    ~user_taskMM()
    {
        delete matrix_A;
        delete matrix_B;
        delete matrix_C;
    }

};

void user_taskMM::init()
{
    int mytrank = getTrank();
    int nthreads = getLsize();
    int local_sizex= size_N/nthreads;
    std::ofstream* output = getThreadOstream();
    *output<<"thread "<<mytrank<<" start initializing sub matrices with random numbers."<<std::endl;
    for(int i=0; i< local_sizex;i++)
    {
        for(int j=0; j< size_N; j++)
        {
            matrix_A[(mytrank*local_sizex + i)*size_N + j] = (float)rand()/size_N;
            matrix_B[(mytrank*local_sizex + i)*size_N + j] = (float)rand()/size_N;

        }
    }

    *output<<"thread "<<mytrank<<" finish initialize."<<std::endl;

}
void user_taskMM::run()
{
    int mytrank = getTrank();
    int nthreads = getLsize();
    int local_sizex= size_N/nthreads;
    std::ofstream* output = getThreadOstream();
    *output<<"thread "<<mytrank<<" start computing: "<<std::endl;
    Timer timer;
    timer.start();
    for(int n= 0; n< nthreads; n++)
    {
        *output<<"\t computing sub MM["<<mytrank<<" "<<n<<"]..."<<std::endl;
        for(int i = 0; i<local_sizex; i++)
        {
            int _x = i+mytrank*local_sizex;
            for(int j =0; j<local_sizex; j++)
            {
                int _y = j+n*local_sizex;
                float tmp = 0.0;
                for(int k =0 ; k<size_N; k++)
                {
                    tmp+=matrix_A[_x*size_N + k]*matrix_B[_y*size_N + k];
                }
                matrix_C[_x*size_N + _y] = tmp;

            }

        }
    }
    double t1 = timer.stop();

    /*for(int i=0; i<local_sizex; i++)
    {
        for(int j=0; j<size_N; j++)
        {
            float tmp =0.0;
            for(int k=0; k<size_N; k++)
            {
                tmp += matrix_A[(i+mytrank*local_sizex)*size_N + k]*
                        matrix_B[j*size_N + k];
            }
            //matrix_C[(i+mytrank*local_sizex)*size_N + j]= tmp;
            if(tmp != matrix_C[(i+mytrank*local_sizex)*size_N + j])
                *output<<"error on"<<(i+mytrank*local_sizex)*size_N<<" "<<j<<std::endl;
        }
    }*/

    *output<<"time cost: "<<t1<<" s"<<std::endl;
}

int main()
{
    UtcContext ctx;
    std::string pname;
    ctx.getProcessorName(pname);
   std::ofstream* pout= getProcOstream();
   *pout<<"proc rank:"<<ctx.getProcRank()<<" processor name:"<<pname.c_str()<<std::endl;

    int t_list[2]={0,0};
    RankList r_list(2, t_list);
    Task<user_taskMM> taskMM("para-MM", r_list);

    Timer timer;
    taskMM.init();

    timer.start();
    taskMM.run();
    double t1=timer.stop();

    taskMM.waitTillDone();
    double t2= timer.stop();

    *pout<<"time1: "<< t1<<" s"<<std::endl;
    *pout<<"time2: "<< t2<<" s"<<std::endl;

    return 0;
}


