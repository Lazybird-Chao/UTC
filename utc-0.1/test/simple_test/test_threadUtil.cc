/*
 * test_threadUtil.cc
 *
 *  Created on: Nov 19, 2016
 *      Author: chao
 */
/* Number of threads to use */
#include "commonUtil.h"
#include "PthreadBarrier.h"
#include "FastBarrier.h"
#include "FastMutex.h"
#include "FastCond.h"
#include "mpi.h"
#include "boost/thread/barrier.hpp"

#define N 2

/* Number of times to call barrier primitive */
#define COUNT 10000

barrier_t bench_b;
FastBarrier bench_fb;
int *bench_buffer;
int buffer_size = 1204;
FastMutex bench_fm;
pthread_mutex_t bench_m;
pthread_cond_t bench_c;
FastCond bench_fc;
bool ready;
boost::barrier *bench_bb;


static void *tbench1(void * arg)
{
	int i;
	double ret = 0;
	for (i = 0; i < COUNT; i++)
	{
		ret = 0;
		for(int j=0; j<100000; j++){
			ret += j*j;
		}
		bench_fb.wait();
	}

	return NULL;
}

static void *tbench2(void *arg){
	int i;
	double ret = 0;
	for (i = 0; i < COUNT; i++)
	{
		ret = 0;
		for(int j=0; j<100000; j++){
			ret += j*j;
		}
		barrier_wait(&bench_b);
	}
	return NULL;
}

static void *tbench7(void *arg){
	int i;
	double ret = 0;
	for (i = 0; i < COUNT; i++)
	{
		ret = 0;
		for(int j=0; j<100000; j++){
			ret += j*j;
		}
		bench_bb->wait();
	}
	return NULL;
}

static void *tbench3(void *arg){
	int i;

	for(int j=0; j< 1000; j++){
		bench_fm.lock();
		for (i = 0; i < buffer_size; i++)
		{
			bench_buffer[i] += i;
		}
		bench_fm.unlock();
		double r;
		for(int i=0; i< 10000; i++)
			r+=i;
	}
	bench_fb.wait();
	return NULL;
}

static void *tbench4(void *arg){
	int i;
	for(int j=0; j< 1000; j++){
		pthread_mutex_lock(&bench_m);
		for (i = 0; i < buffer_size; i++)
		{
			bench_buffer[i] += i;
		}
		pthread_mutex_unlock(&bench_m);
		double r;
		for(int i=0; i< 10000; i++)
			r+=i;
	}
	barrier_wait(&bench_b);
	return NULL;
}

static void *tbench5(void *arg){
	int i;
	double r;
	for (i = 0; i < 100000; i++)
	{
		r += i;
	}
	bench_fm.lock();
	while(ready==false){
		bench_fc.wait(&bench_fm);
	}
	bench_fm.unlock();

	bench_fb.wait();
	return NULL;
}

static void *tbench6(void *arg){
	int i;
	double r;
	for (i = 0; i < 100000; i++)
	{
		r += i;
	}
	pthread_mutex_lock(&bench_m);
	while(ready ==false){
		pthread_cond_wait(&bench_c,&bench_m);
	}
	pthread_mutex_unlock(&bench_m);
	barrier_wait(&bench_b);
	return NULL;
}



int main(int argc, char* argv[])
{
	unsigned int n = N;
	if(argc>1)
		n = atoi(argv[1]);
	pthread_t th[n - 1];
	int i;

	printf("start test fast barrier\n");
	bench_fb.init(n);
	for (i = 0; i < n - 1; i++)
	{
		if (pthread_create(&th[i], NULL, tbench1, NULL))
		{
			errx(1, "pthread_create failed");
		}
	}
	double t1= MPI_Wtime();
	tbench1(NULL);
	double t2 = MPI_Wtime();
	printf("time: %f\n", t2-t1);
	printf("end test fast barrier\n\n");
	for (i = 0; i < n - 1; i++)
	{
		if (pthread_join(th[i], NULL)) errx(1, "pthread_join failed");
	}


	printf("start test pthread barrier\n");
	barrier_init(&bench_b, n);
	for (i = 0; i < n - 1; i++)
	{
		if (pthread_create(&th[i], NULL, tbench2, NULL))
		{
			errx(1, "pthread_create failed");
		}
	}
	t1= MPI_Wtime();
	tbench2(NULL);
	t2 = MPI_Wtime();
	printf("time: %f\n", t2-t1);
	//barrier_destroy(&bench_b);
	printf("end test pthread barrier\n\n");
	for (i = 0; i < n - 1; i++)
	{
		if (pthread_join(th[i], NULL)) errx(1, "pthread_join failed");
	}

	printf("start test boost barrier\n");
	bench_bb = new boost::barrier(n);
	for (i = 0; i < n - 1; i++)
	{
		if (pthread_create(&th[i], NULL, tbench7, NULL))
		{
			errx(1, "pthread_create failed");
		}
	}
	t1= MPI_Wtime();
	tbench7(NULL);
	t2 = MPI_Wtime();
	printf("time: %f\n", t2-t1);
	//barrier_destroy(&bench_b);
	printf("end test boost barrier\n\n");
	for (i = 0; i < n - 1; i++)
	{
		if (pthread_join(th[i], NULL)) errx(1, "pthread_join failed");
	}

/*
	printf("start test fast mutex\n");
	bench_buffer = (int*)malloc(buffer_size*sizeof(int));
	for( i=0; i<buffer_size; i++)
		bench_buffer[i]= 0;
	for (i = 0; i < n - 1; i++)
	{
		if (pthread_create(&th[i], NULL, tbench3, NULL))
		{
			errx(1, "pthread_create failed");
		}
	}
	t1= MPI_Wtime();
	tbench3(NULL);
	t2 = MPI_Wtime();
	printf("time: %f\n", t2-t1);
	printf("end test fast mutex\n\n");
	for (i = 0; i < n - 1; i++)
	{
		if (pthread_join(th[i], NULL)) errx(1, "pthread_join failed");
	}
	for(i=0; i<buffer_size; i++){
		if(bench_buffer[i] != n*i*1000)
			printf("error with fast mutex\n");
	}


	printf("start test pthread mutex\n");
	for(int i=0; i<buffer_size; i++)
		bench_buffer[i]= 0;
	pthread_mutex_init(&bench_m, NULL);
	for (i = 0; i < n - 1; i++)
	{
		if (pthread_create(&th[i], NULL, tbench4, NULL))
		{
			errx(1, "pthread_create failed");
		}
	}
	t1= MPI_Wtime();
	tbench4(NULL);
	t2 = MPI_Wtime();
	printf("time: %f\n", t2-t1);
	printf("end test pthread mutex\n\n");
	for (i = 0; i < n - 1; i++)
	{
		if (pthread_join(th[i], NULL)) errx(1, "pthread_join failed");
	}
	for(i=0; i<buffer_size; i++){
		if(bench_buffer[i] != n*i*1000)
			printf("error with pthread mutex\n");
	}


	printf("start test fast cond\n");
	ready = false;
	for (i = 0; i < n - 1; i++)
	{
		if (pthread_create(&th[i], NULL, tbench5, NULL))
		{
			errx(1, "pthread_create failed");
		}
	}
	t1= MPI_Wtime();
	double r=0;
	for(int i=0; i<4000000; i++)
		r+=i;
	bench_fm.lock();
	ready = true;
	bench_fm.unlock();
	bench_fc.broadcast();
	bench_fb.wait();
	t2 = MPI_Wtime();
	printf("time: %f\n", t2-t1);
	printf("end test fast cond\n\n");


	printf("start test pthread cond\n");
	ready = false;
	for (i = 0; i < n - 1; i++)
	{
		if (pthread_create(&th[i], NULL, tbench6, NULL))
		{
			errx(1, "pthread_create failed");
		}
	}
	t1= MPI_Wtime();
	r=0;
	for(int i=0; i<4000000; i++)
		r+=i;
	pthread_mutex_lock(&bench_m);
	ready = true;
	pthread_mutex_unlock(&bench_m);
	pthread_cond_broadcast(&bench_c);
	barrier_wait(&bench_b);
	t2 = MPI_Wtime();
	printf("time: %f\n", t2-t1);
	printf("end test pthread cond\n\n");

*/
	puts("bench OK");

	return 0;
}



