/*
 * Pause_precision_test.cc
 *
 *  Created on: Mar 25, 2016
 *      Author: chao
 */
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <xmmintrin.h>
#include <thread>
#include <mpi.h>

int main(int argc, char*argv[]){
	struct timespec delay;
	delay.tv_sec=0;
	delay.tv_nsec=100000;
	double t1 = MPI_Wtime();
	for(long i=0; i<10;i++)
	//while(1)
{
		nanosleep(&delay, NULL);
		//_mm_pause();
		//__asm__ __volatile__("pause");
		//std::this_thread::yield();
 }
	double t2 = MPI_Wtime();
	printf("%.6f\n", (t2-t1)*1.0e6);
	return 0;
}



