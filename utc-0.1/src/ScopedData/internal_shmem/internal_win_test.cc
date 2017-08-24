/*
 * test.cc
 *
 *  Created on: Aug 23, 2017
 *      Author: Chao
 */

#include <mpi.h>
#include <unistd.h>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include "internal_win.h"
#include "scoped_shmem.h"

int main(int argc, char**argv){
	MPI_Comm worldcomm = MPI_COMM_WORLD;
	int mpi_provided;
	MPI_Init_thread( &argc, &argv, MPI_THREAD_SERIALIZED, &mpi_provided );

	int rank, size;
	MPI_Comm_rank( MPI_COMM_WORLD, &rank );
	MPI_Comm_size( MPI_COMM_WORLD, &size );
	std::cout<<"global rank/size: "<<rank<<" "<<size<<std::endl;

	int color = rank/3;

	MPI_Comm subcomm;
	MPI_Comm_split(MPI_COMM_WORLD, color, rank, &subcomm);

	int lrank, lsize;
	MPI_Comm_rank( subcomm, &lrank );
	MPI_Comm_size( subcomm, &lsize );
	std::cout<<"local rank/size: "<<lrank<<" "<<lsize<<std::endl;

	iUtc::internal_MPIWin globalWin(&worldcomm, 1024, 0);
	iUtc::internal_MPIWin localWin(&subcomm, 1024, 0);

	scoped_shmem_init_comm(globalWin);
	scoped_shmem_init_comm(localWin);

	int *localshare = (int*)scoped_shmem_malloc(64, localWin);
	int *globalshare = (int*)scoped_shmem_malloc(64, globalWin);

	if(rank==3){
		char message[100];
		char message2[100];
		strcpy(message, "I am global rank with 2");
		scoped_shmem_char_put((char*)localshare, message, 100, 1, localWin);
		scoped_shmem_char_put((char*)globalshare, message, 100, 1, globalWin);
	}
	scoped_shmem_barrier(localWin);
	scoped_shmem_barrier(globalWin);

	if(lrank==1){
		std::cout<<rank<<"-"<<lrank<<"(local): "<<(char*)localshare<<std::endl;
		std::cout<<rank<<"-"<<lrank<<"(global): "<<(char*)globalshare<<std::endl;
	}


	MPI_Finalize();
	return 0;

}



