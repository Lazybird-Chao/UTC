/*
 * mpi_bw.cc
 *
 */

#include <mpi.h>
#include <stdlib.h>
#include <iostream>
#include <cstdio>
#include <cmath>
#include <unistd.h>


#define MAX_ALIGNMENT 65536
#define MAX_MSG_SIZE (1<<27)
#define MYBUFSIZE (MAX_MSG_SIZE + MAX_ALIGNMENT)

#define LOOP_LARGE 20
#define WINDOW_SIZE_LARGE 64
#define SKIP_LARGE 2
#define LARGE_MESSAGE_SIZE (1<<16)
#define LOOP_LLARGE 10
#define SKIP_LLARGE 1
#define LLARGE_MESSAGE_SIZE (1<<23)
#define WINDOW_SIZE_LLARGE 4

int main(int argc, char*argv[])
{
	int myid, numprocs, i, j;
	int size;
	char *s_buf, *r_buf;
	double t_start=0.0, t_end=0.0, t=0.0;
	int loop=100;
	int window_size=64;
	int skip=10;
	MPI_Request *request = (MPI_Request*)malloc(sizeof(MPI_Request)*window_size);
	MPI_Status *reqstat = (MPI_Status*)malloc(sizeof(MPI_Status)*window_size);


	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	if(numprocs %2 !=0)
	{
		printf("This test requires even processes\n");
		MPI_Finalize();
		return 1;
	}
	int align_size = getpagesize();
	char* buf_original = (char*)malloc(MYBUFSIZE);
	if(myid % 2 ==0)
	{
		s_buf = (char *) (((unsigned long) buf_original + (align_size - 1)) /
				align_size * align_size);
		for(i =0; i<MAX_MSG_SIZE; i++)
			s_buf[i]='s';
		r_buf = (char*)malloc(4);
	}
	else if(myid %2 ==1)
	{
		r_buf = (char *) (((unsigned long) buf_original + (align_size - 1)) /
				align_size * align_size);
		s_buf = (char*)malloc(4);
		for(i=0; i<4; i++)
			s_buf[i] ='r';
	}

	// bandwidth test
	if(!myid)
		printf("Size\t\tMbit/s\n");
	for(size =1; size<=MAX_MSG_SIZE; size*=2)
	{
		if(size>LARGE_MESSAGE_SIZE)
		{
			loop = LOOP_LARGE;
			skip = SKIP_LARGE;
			window_size = WINDOW_SIZE_LARGE;
		}
		else if(size>LLARGE_MESSAGE_SIZE)
		{
			loop=LOOP_LLARGE;
			skip=SKIP_LLARGE;
			window_size = WINDOW_SIZE_LLARGE;
		}

		if(myid %2==0)
		{
			for(i=0; i<loop+skip; i++)
			{
				if(i==skip)
					t_start= MPI_Wtime();
				for(j=0; j<window_size; j++)
					MPI_Isend(s_buf, size, MPI_CHAR, myid+1, i*window_size+j, MPI_COMM_WORLD, request+j);
				for(j=0; j<window_size; j++)
					MPI_Wait(request+j, reqstat+j);
				MPI_Recv(r_buf, 4, MPI_CHAR, myid+1, i*window_size+j, MPI_COMM_WORLD, &reqstat[0]);
			}
			t_end = MPI_Wtime();
			t = t_end - t_start;
			double bw = size/1e6*loop*window_size*8;
			bw = bw/t;
			printf("%d\t\t%f\t\t%d\n", size, bw, myid);
		}
		else if(myid %2 == 1)
		{
			for(i=0; i<loop+skip; i++)
			{
				for(j=0; j<window_size; j++)
					MPI_Irecv(r_buf, size, MPI_CHAR, myid-1, i*window_size+j, MPI_COMM_WORLD, request+j);
				for(j=0; j<window_size; j++)
					MPI_Wait(request+j, reqstat+j);
				MPI_Send(r_buf, 4, MPI_CHAR, myid-1, i*window_size+j, MPI_COMM_WORLD);
			}
		}

	}

	MPI_Finalize();
	return 0;

}



