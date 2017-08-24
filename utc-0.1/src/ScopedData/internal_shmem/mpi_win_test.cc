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

	/*iUtc::internal_MPIWin globalWin(&worldcomm, 1024, 0);
	iUtc::internal_MPIWin localWin(&subcomm, 1024, 0);

	scoped_shmem_init_comm(globalWin);
	scoped_shmem_init_comm(localWin);*/

	MPI_Win localWin;
	MPI_Win globalWin;
	char localBase[1024];
	char globalBase[1024];
	MPI_Win localWin2;
	char localBase2[1024];
	MPI_Win_create(localBase, 1024, 1, MPI_INFO_NULL, subcomm, &localWin);
	MPI_Win_create(localBase2, 1024, 1, MPI_INFO_NULL, subcomm, &localWin2);
	MPI_Win_create(globalBase, 1024, 1, MPI_INFO_NULL, worldcomm, &globalWin);
	if(lrank==0 && rank == 3){
		MPI_Win_lock(MPI_LOCK_SHARED,1,1,localWin);
		//MPI_Win_lock(MPI_LOCK_SHARED,1,1,localWin2);
		MPI_Win_lock(MPI_LOCK_SHARED,1,1,globalWin);
		char message[100];
		strcpy(message, "I am global rank with 2");
		MPI_Put(message, 100, MPI_CHAR, 1, 0, 100, MPI_CHAR, localWin);
		//MPI_Put(message, 100, MPI_CHAR, 1, 0, 100, MPI_CHAR, localWin2);
		MPI_Put(message, 100, MPI_CHAR, 1, 0, 100, MPI_CHAR, globalWin);
	}
	if(lrank==2 && rank==5){
		MPI_Win_lock(MPI_LOCK_SHARED,1,1,localWin);
		char message[100];
		sleep(1);
		strcpy(message, "I am global rank with 5");
		MPI_Put(message, 100, MPI_CHAR, 1, 0, 100, MPI_CHAR, localWin);
	}
	if(rank==3){
		MPI_Win_flush(1, localWin);
		//MPI_Win_flush(1, localWin2);
		MPI_Win_flush(1, globalWin);
	}
	if(rank==5){
		MPI_Win_flush(1, localWin);
	}
	MPI_Barrier(worldcomm);
	if(lrank==1){
		std::cout<<rank<<"-"<<lrank<<"(local): "<<localBase<<std::endl;
		std::cout<<rank<<"-"<<lrank<<"(local): "<<localBase+100<<std::endl;
		std::cout<<rank<<"-"<<lrank<<"(global): "<<globalBase<<std::endl;
	}

	if(rank == 3){
		MPI_Win_unlock(1, localWin);
		//MPI_Win_unlock(1, localWin2);
		MPI_Win_unlock(1, globalWin);
	}
	if(rank==5){
		MPI_Win_unlock(1, localWin);
	}
	MPI_Win_free(&localWin);
	MPI_Win_free(&globalWin);
	MPI_Win_free(&localWin2);


	MPI_Finalize();
	return 0;

}



