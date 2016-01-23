/*
 * mpi_latency.cc
 *
 *
 */

#include <mpi.h>
#include <iostream>
#include <string>
#include <cstdio>
#include <cstring>
#include <stdlib.h>

#define MESSAGE_ALIGNMENT 64
#define MAX_MSG_SIZE (1<<27)
#define MYBUFSIZE (MAX_MSG_SIZE + MESSAGE_ALIGNMENT)
#define SKIP_LARGE  10
#define SKIP_LLARGE 1
#define LOOP_LARGE  100
#define LOOP_LLARGE 10
#define LARGE_MESSAGE_SIZE  8192
#define LLARGE_MESSAGE_SIZE (8192*1024)
int skip = 1000;
int loop = 10000;

int main(int argc, char* argv[])
{
	int myid, numprocs, i;
    int size;
    MPI_Status reqstat;
    char *s_buf, *r_buf;
    int align_size=MESSAGE_ALIGNMENT;
    double t_start = 0.0, t_end = 0.0;
    //char end_data;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    if(numprocs != 2)
    {
        if(myid == 0)
            std::cerr<<"This test requires exactly two processes\n";
		MPI_Finalize();
		return 1;
    }


    if(myid == 0)
    {
    	char* s_buf_tmp = (char*)malloc(MYBUFSIZE);
    	s_buf = (char*)(((unsigned long)s_buf_tmp + (MESSAGE_ALIGNMENT-1)) / MESSAGE_ALIGNMENT*MESSAGE_ALIGNMENT);
    	char* r_buf_tmp = (char*)malloc(MYBUFSIZE);
    	r_buf = (char*)(((unsigned long)s_buf_tmp + (MESSAGE_ALIGNMENT-1)) / MESSAGE_ALIGNMENT*MESSAGE_ALIGNMENT);
    	for(int i =0; i<MAX_MSG_SIZE; i++)
		{
			s_buf[i]='s';
		}
		std::cout<<"Byte"<<"\t\t"<<"us"<<std::endl;
		for(size=0; size<=MAX_MSG_SIZE; size= (size?size*2:1))
		{
			if(size > LARGE_MESSAGE_SIZE)
			{
				loop = LOOP_LARGE;
				skip = SKIP_LARGE;
			}
			else if(size > LLARGE_MESSAGE_SIZE)
			{
				loop = LOOP_LLARGE;
				skip = SKIP_LLARGE;
			}
			int i =0;
			for(i=0; i<loop+skip; i++)
			{
				if(i==skip)
					t_start = MPI_Wtime();

				MPI_Send(s_buf, size, MPI_CHAR, 1, i, MPI_COMM_WORLD);
				MPI_Recv(r_buf, size, MPI_CHAR, 1, i, MPI_COMM_WORLD, &reqstat);
			}
			//MPI_Recv(&end_data, 1, MPI_CHAR, 1, i, MPI_COMM_WORLD, &reqstat);
			t_end = MPI_Wtime();

			double latency = (t_end - t_start) * 1e6 / (2*loop);
			std::cout<<size<<"\t\t"<<latency<<std::endl;
		}
		free(s_buf_tmp);
		free(r_buf_tmp);
    }
    else if(myid == 1)
    {
    	char* r_buf_tmp = (char*)malloc(MYBUFSIZE);
		r_buf = (char*)(((unsigned long)r_buf_tmp + (MESSAGE_ALIGNMENT-1)) / MESSAGE_ALIGNMENT*MESSAGE_ALIGNMENT);
		char* s_buf_tmp = (char*)malloc(MYBUFSIZE);
		s_buf = (char*)(((unsigned long)r_buf_tmp + (MESSAGE_ALIGNMENT-1)) / MESSAGE_ALIGNMENT*MESSAGE_ALIGNMENT);
		for(int i =0; i<MAX_MSG_SIZE; i++)
		{
			s_buf[i]='r';
		}
		//end_data = 'e';

		for(size=0; size<=MAX_MSG_SIZE; size= (size?size*2:1))
		{
			if(size > LARGE_MESSAGE_SIZE)
			{
				loop = LOOP_LARGE;
				skip = SKIP_LARGE;
			}
			int i =0;
			for(i=0; i<loop+skip; i++)
			{
				MPI_Recv(r_buf, size, MPI_CHAR, 0, i, MPI_COMM_WORLD, &reqstat);
				MPI_Send(s_buf, size, MPI_CHAR, 0, i, MPI_COMM_WORLD);
			}
			//MPI_Send(&end_data, 1, MPI_CHAR, 0, i, MPI_COMM_WORLD);

		}
		free(r_buf_tmp);
		free(s_buf_tmp);
    }

    MPI_Finalize();

}


