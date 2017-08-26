/*
 * test.cc
 *
 *  Created on: Aug 23, 2017
 *      Author: Chao
 */

#include <sys/types.h>
#include <unistd.h>

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
	std::cout<<"global rank/size: "<<rank<<" "<<size<<" pid "<<getpid()<<std::endl;

	int color = rank/3;

	MPI_Comm subcomm;
	MPI_Comm_split(MPI_COMM_WORLD, color, rank, &subcomm);

	int lrank, lsize;
	MPI_Comm_rank( subcomm, &lrank );
	MPI_Comm_size( subcomm, &lsize );
	std::cout<<"local rank/size: "<<lrank<<" "<<lsize<<std::endl;

	iUtc::internal_MPIWin globalWin(&worldcomm, 131072, 0);
	iUtc::internal_MPIWin localWin(&subcomm, 131072, 0);

	MPI_Barrier(worldcomm);

	scoped_shmem_init_comm(globalWin);
	scoped_shmem_init_comm(localWin);

	sleep(rank);
	std::cout<<rank<<" localWin info: "<<localWin.get_scoped_win_comm_rank()<<" "
			<<localWin.get_scoped_win_comm_size()<<" "
			<<localWin.get_heap_mspace()<<" "
			<<localWin.get_heap_base_address()<<std::endl;

	MPI_Barrier(worldcomm);

	sleep(rank);
	int *localshare = (int*)scoped_shmem_malloc(128, localWin);
	std::cout<<rank<<" localsharespace info: "
			<<localshare<<" "
			<<scoped_shmem_addr_offset(localshare,localWin)
			<<std::endl;
	int *globalshare = (int*)scoped_shmem_malloc(128, globalWin);
	std::cout<<rank<<" globalsharespace info: "
				<<globalshare<<" "
				<<scoped_shmem_addr_offset(globalshare,globalWin)
				<<std::endl;
	MPI_Barrier(worldcomm);

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

	MPI_Barrier(worldcomm);
	sleep(rank);
	int *globalshare2 = (int*)scoped_shmem_malloc(100, globalWin);
	std::cout<<rank<<" globalsharespace2 info: "
					<<globalshare<<" "
					<<scoped_shmem_addr_offset(globalshare2,globalWin)
					<<std::endl;

	MPI_Barrier(worldcomm);
	sleep(rank);
	if(color==1){
		int *localshare2 = (int*)scoped_shmem_malloc(100, localWin);
		std::cout<<rank<<" localsharespace2 info: "
						<<globalshare<<" "
						<<scoped_shmem_addr_offset(localshare2,localWin)
						<<std::endl;
	}

	scoped_shmem_free(localshare, localWin);
	scoped_shmem_free(globalshare, globalWin);

	MPI_Barrier(worldcomm);
	sleep(rank);
	if(color==0){
		int *localshare3 = (int*)scoped_shmem_malloc(400, localWin);
		std::cout<<rank<<" localsharespace3 info: "
						<<globalshare<<" "
						<<scoped_shmem_addr_offset(localshare3,localWin)
						<<std::endl;
	}

	MPI_Barrier(worldcomm);
	sleep(rank);
	int *globalshare3 = (int*)scoped_shmem_malloc(1000, globalWin);
	std::cout<<rank<<" globalsharespace3 info: "
					<<globalshare<<" "
					<<scoped_shmem_addr_offset(globalshare3,globalWin)
					<<std::endl;

	MPI_Barrier(worldcomm);
	scoped_shmem_free(globalshare2, globalWin);

	sleep(1);
	if(color==0)
		int *localshare4 = (int*)scoped_shmem_malloc(666, localWin);
	MPI_Barrier(worldcomm);

	sleep(1);
	int *localshare5 = (int*)scoped_shmem_malloc(1024, localWin);
	MPI_Barrier(worldcomm);

	sleep(1);
	int *globalshare4 = (int*)scoped_shmem_malloc(2048, globalWin);


	scoped_shmem_finalize_comm(globalWin);
	scoped_shmem_finalize_comm(localWin);
	MPI_Barrier(worldcomm);

	MPI_Finalize();
	return 0;

}



