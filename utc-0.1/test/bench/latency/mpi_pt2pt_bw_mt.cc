/*
 * mpi_pt2pt_bw_mt.c
 *
 *  Created on: Oct 9, 2017
 *      Author: Chao
 */

#include <pthread.h>
#include <iostream>
#include <cstdlib>

#include "mpi.h"


#define ITERS_SMALL     (500)
#define ITERS_LARGE     (50)
#define LARGE_THRESHOLD (8192)
#define MAX_MSG_SZ (1<<22)
#define LOOP (10)
#define MAX_NUM_THREADS 64

#define MESSAGE_ALIGNMENT (1<<12)
#define MYBUFSIZE (MAX_MSG_SZ * ITERS_LARGE + MESSAGE_ALIGNMENT)

pthread_mutex_t finished_size_mutex;
pthread_cond_t  finished_size_cond;
int counter=0;

typedef struct {
	int tid;
	int pid;
	int nthreads;
	double ** bw;
} thread_arg_t;

void * send_thread(void* arg);
void * recv_thread(void* arg);

int main(int argc, char *argv[]){
	int numprocs, provided, myid, err;
	int i = 0;
	pthread_t sr_threads[MAX_NUM_THREADS];

	pthread_mutex_init(&finished_size_mutex, NULL);
	pthread_cond_init(&finished_size_cond, NULL);

	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	if(numprocs != 2) {
		if(myid == 0) {
			std::cout<< "This test requires exactly two processes\n";
		}

		MPI_Finalize();

		return 1;
	}

	int nthreads = 1;
	if(argc > 1){
		nthreads = atoi(argv[1]);
	}
	double **bandrecord = new double*[64];
	for(int i = 0; i<64; i++)
		bandrecord[i] = new double[23];

	if(myid == 0) {
		for(int i = 0; i<nthreads; i++){
			thread_arg_t *arg = new thread_arg_t;
			arg->bw = (double **)bandrecord;
			arg->tid = i;
			arg->pid = myid;
			arg->nthreads = nthreads;
			pthread_create(&sr_threads[i], NULL, send_thread, (void*)arg);
		}
		for(int i = 0; i<nthreads; i++)
			pthread_join(sr_threads[i], NULL);

	}

	else {
		for(i = 0; i < nthreads; i++) {
			thread_arg_t *arg = new thread_arg_t;
			arg->bw = (double**)bandrecord;
			arg->tid = i;
			arg->pid = myid;
			arg->nthreads = nthreads;
			pthread_create(&sr_threads[i], NULL, recv_thread, (void*)arg);
		}

		for(i = 0; i < nthreads; i++) {
			pthread_join(sr_threads[i], NULL);
		}
	}

	if(myid == 0){
		std::cout<<"average message: \n";
		double avg_mr[23];
		for(int i = 0; i<23; i++){
			avg_mr[i] = 0;
			for(int j = 0; j<nthreads; j++)
				avg_mr[i] += bandrecord[j][i];
			//avg_mr[i] /= ntasks;
			std::cout<<(1<<i)<<"\t\t"<<avg_mr[i]<<std::endl;
		}
	}
	MPI_Finalize();
	return 0;

}


void * send_thread(void * arg) {
	thread_arg_t * targ = (thread_arg_t*)arg;
	int tid = targ->tid;
	int pid = targ->pid;
	double** bandwidth = targ->bw;
	int totalthreads = targ->nthreads;
	MPI_Request *request = (MPI_Request*)malloc(sizeof(MPI_Request)*ITERS_SMALL);
	MPI_Status *reqstat = (MPI_Status*)malloc(sizeof(MPI_Status)*ITERS_SMALL);

	char *localdataBuff = new char[MYBUFSIZE];
	for(int i = 0; i<MYBUFSIZE; i++){
		localdataBuff[i] = (char)('0'+(tid + MYBUFSIZE)%10);
	}
	localdataBuff = (char*)(((unsigned long)localdataBuff + (MESSAGE_ALIGNMENT-1))
					/MESSAGE_ALIGNMENT*MESSAGE_ALIGNMENT);


	pthread_mutex_lock(&finished_size_mutex);
	counter++;
	if(counter == totalthreads){
		pthread_mutex_unlock(&finished_size_mutex);
		pthread_cond_broadcast(&finished_size_cond);
	}else{
		pthread_cond_wait(&finished_size_cond, &finished_size_mutex);
		pthread_mutex_unlock(&finished_size_mutex);
	}


	int count = 0;
	for(int size = 1; size <= MAX_MSG_SZ; size <<= 1){
		double t_start = MPI_Wtime();
		int iters = size < LARGE_THRESHOLD ? ITERS_SMALL : ITERS_LARGE;
		if(pid == 0){
			for(int j = 0; j<LOOP; j++){
				for(int i = 0, offset = 0; i<iters; i++, offset+=size){
					MPI_Isend(localdataBuff+offset, size, MPI_CHAR, 1, (tid<<10)+i, MPI_COMM_WORLD, request+i);
				}
				for(int i = 0; i<iters; i++)
					MPI_Wait(request+i, reqstat+i);
				MPI_Recv(localdataBuff, 4, MPI_CHAR, 1, (tid<<10)+iters, MPI_COMM_WORLD, &reqstat[0]);
			}
		}

		double t_end = MPI_Wtime();
		double t = t_end - t_start;
		double bw = (double)size/1e6*iters*LOOP;
		bw = bw/t;
		if(pid ==0 && tid ==0){
			std::cout<<size<<"\t\t"<<bw<<std::endl;
		}
		if(pid == 0){
			bandwidth[tid][count] = bw;
		}
		count++;
	}


    return 0;
}


void * recv_thread(void * arg) {
	thread_arg_t * targ = (thread_arg_t*)arg;
	int tid = targ->tid;
	int pid = targ->pid;
	double** bandwidth = targ->bw;
	int totalthreads = targ->nthreads;

	MPI_Request *request = (MPI_Request*)malloc(sizeof(MPI_Request)*ITERS_SMALL);
	MPI_Status *reqstat = (MPI_Status*)malloc(sizeof(MPI_Status)*ITERS_SMALL);

	char *localdataBuff = new char[MYBUFSIZE];
	for(int i = 0; i<MYBUFSIZE; i++){
		localdataBuff[i] = (char)('0'+(tid + MYBUFSIZE)%10);
	}
	localdataBuff = (char*)(((unsigned long)localdataBuff + (MESSAGE_ALIGNMENT-1))
					/MESSAGE_ALIGNMENT*MESSAGE_ALIGNMENT);


	pthread_mutex_lock(&finished_size_mutex);
	counter++;
	if(counter == totalthreads){
		pthread_mutex_unlock(&finished_size_mutex);
		pthread_cond_broadcast(&finished_size_cond);
	}else{
		pthread_cond_wait(&finished_size_cond, &finished_size_mutex);
		pthread_mutex_unlock(&finished_size_mutex);
	}


	for(int size = 1; size <= MAX_MSG_SZ; size <<= 1){
		int iters = size < LARGE_THRESHOLD ? ITERS_SMALL : ITERS_LARGE;
		if(pid == 1){
			for(int j = 0; j<LOOP; j++){
				for(int i = 0, offset = 0; i<iters; i++, offset+=size){
					MPI_Irecv(localdataBuff+offset, size, MPI_CHAR, 0, (tid<<10)+i, MPI_COMM_WORLD, request+i);
				}
				for(int i = 0; i<iters; i++)
					MPI_Wait(request+i, reqstat+i);
				MPI_Send(localdataBuff, 4, MPI_CHAR, 0, (tid<<10)+iters, MPI_COMM_WORLD);
			}
		}
	}


	return 0;
}
