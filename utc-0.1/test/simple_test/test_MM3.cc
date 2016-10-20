/*
 * sequential matrix multiply
 */
#include "Utc.h"
#include <iostream>
#include <cstdlib>

#define size_N 1024

int main()
{
    float *matrix_A = new float[size_N*size_N];
    float *matrix_B = new float[size_N*size_N];
    float *matrix_C = new float[size_N*size_N];

    std::cout<<"initialize matrices."<<std::endl;
    for(int i =0; i<size_N; i++)
    {
        for(int j=0; j<size_N; j++)
        {
            matrix_A[i*size_N + j] = (float)rand()/size_N;
            matrix_B[i*size_N + j] = (float)rand()/size_N;
        }
    }

    std::cout<<"start computing."<<std::endl;
    iUtc::Timer timer;
    timer.start();
    for(int i=0; i<size_N; i++)
    {
        for(int j=0; j<size_N; j++)
        {
            float tmp = 0.0;
            for(int k=0; k<size_N; k++)
            {
                tmp = matrix_A[i*size_N + k]*matrix_B[j*size_N +k];
            }
            matrix_C[i*size_N + j]= tmp;
        }
    }
    double t = timer.stop();
    std::cout<<"time cost: "<<t<<" s"<<std::endl;
    delete matrix_A;
    delete matrix_B;
    delete matrix_C;
    return 0;
}
