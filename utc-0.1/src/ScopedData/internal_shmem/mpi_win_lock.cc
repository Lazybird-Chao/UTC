/*
 * mpi_win_lock.cc
 *
 *  Created on: Dec 12, 2016
 *      Author: chao
 */

#include "mpi_win_lock.h"
#include <cstdlib>

#define NEXT_DISP 1
#define PREV_DISP 0
#define TAIL_DISP 2 //Only has meaning in root
#define LOCK_DISP 3

namespace iUtc{

MpiWinLock::MpiWinLock(){
	m_root = 0;
	world_comm_copy = MPI_COMM_WORLD;
	m_comm = &world_comm_copy;
	MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);

	MPI_Info lock_info = MPI_INFO_NULL;
	MPI_Info_create(&lock_info);

	MPI_Info_set(lock_info,"same_size","true");

	/*
	MPI_Win_allocate(4*sizeof(int), sizeof(int), lock_info,
			*m_comm, &m_mpi_win_baseptr, &m_mpi_win);
	*/
	m_mpi_win_baseptr = (int*)malloc(4*sizeof(int));
	MPI_Win_create(m_mpi_win_baseptr, 4*sizeof(int), 1,
			lock_info, *m_comm, &m_mpi_win);
	m_mpi_win_baseptr[NEXT_DISP] = -1;
	m_mpi_win_baseptr[PREV_DISP] = -1;
	m_mpi_win_baseptr[TAIL_DISP] = -1;
	m_mpi_win_baseptr[LOCK_DISP] = -1;
	MPI_Win_lock_all (0, m_mpi_win);	//note the 0

	MPI_Info_free(&lock_info);
}


MpiWinLock::MpiWinLock(MPI_Comm *comm, int root, int rank){
	m_root = root;
	m_comm = comm;
	m_rank = rank;

	MPI_Info lock_info = MPI_INFO_NULL;
	MPI_Info_create(&lock_info);

	MPI_Info_set(lock_info,"same_size","true");


	/*MPI_Win_allocate(4*sizeof(int), sizeof(int), lock_info,
			*m_comm, &m_mpi_win_baseptr, &m_mpi_win);
	*/
	m_mpi_win_baseptr = (int*)malloc(4*sizeof(int));
	MPI_Win_create(m_mpi_win_baseptr, 4*sizeof(int), 1,
			lock_info, *m_comm, &m_mpi_win);
	m_mpi_win_baseptr[NEXT_DISP] = -1;
	m_mpi_win_baseptr[PREV_DISP] = -1;
	m_mpi_win_baseptr[TAIL_DISP] = -1;
	m_mpi_win_baseptr[LOCK_DISP] = -1;
	MPI_Win_lock_all (0, m_mpi_win);	//note the 0, not shared lock

	MPI_Info_free(&lock_info);
}

MpiWinLock::~MpiWinLock(){
	MPI_Win_unlock_all(m_mpi_win);
	MPI_Win_free(&m_mpi_win);
	free(m_mpi_win_baseptr);
}


void MpiWinLock::lock(long *lockp){
	MPI_Status status;
	mpi_win_lock_t *lock = (mpi_win_lock_t *) lockp;

	m_mpi_win_baseptr[LOCK_DISP] = 1;
	/* Replace myself with the last tail */
	MPI_Fetch_and_op (&m_rank, &(lock->prev_owner_rank), MPI_INT, m_root,
			TAIL_DISP, MPI_REPLACE, m_mpi_win);
	MPI_Win_flush (m_root, m_mpi_win);

	/* Previous proc holding lock will eventually notify */
	if (lock->prev_owner_rank != -1)
	{
	  /* Send my shmem_world_rank to previous proc's next */
	  MPI_Accumulate (&m_rank, 1, MPI_INT, lock->prev_owner_rank, NEXT_DISP,
			  1, MPI_INT, MPI_REPLACE, m_mpi_win);
	  MPI_Win_flush (lock->prev_owner_rank, m_mpi_win);
	  MPI_Probe (lock->prev_owner_rank, MPI_ANY_TAG, *m_comm, &status);
	}
	/* Hold lock */
	//m_mpi_win_baseptr[LOCK_DISP] = 1;
	MPI_Win_sync (m_mpi_win);

	return;
}


void MpiWinLock::unlock(long *lockp){
	mpi_win_lock_t *lock = (mpi_win_lock_t *) lockp;
	int resettaill = -1;
	int pre;
	/* check if other proc has came to get lock*/
	MPI_Compare_and_swap (&resettaill, &m_rank, &pre, MPI_INT,
				m_root, TAIL_DISP, m_mpi_win);
	MPI_Win_flush (m_rank, m_mpi_win);
	if(pre != m_rank){
		/* there are some one else do the lock op,
		 * so we should get a next_owner_rank */
		while(1){
			MPI_Fetch_and_op (NULL, &(lock->next_owner_rank), MPI_INT, m_rank,
					NEXT_DISP, MPI_NO_OP, m_mpi_win);
			MPI_Win_flush (m_rank, m_mpi_win);
			if (lock->next_owner_rank != -1)
			{
				MPI_Send (&m_rank, 1, MPI_INT, lock->next_owner_rank, -1, *m_comm);
				break;
			}
		}
	}
	else{
		/* no other do the lock op till now, we can just leave, no need to
		 * do notify, also we have reset the taill value in root to -1*/
	}

	/* Determine my next process */
	/*MPI_Fetch_and_op (NULL, &(lock->next_owner_rank), MPI_INT, m_rank,
			NEXT_DISP, MPI_NO_OP, m_mpi_win);
	MPI_Win_flush (m_rank, m_mpi_win);

	if (lock->next_owner_rank != -1)
	{
		MPI_Send (&m_rank, 1, MPI_INT, lock->next_owner_rank, -1, *m_comm);
	}
	*/
	/* Release lock */
	m_mpi_win_baseptr[LOCK_DISP] = -1;
	MPI_Win_sync (m_mpi_win);

	return;
}


int MpiWinLock::trylock(long *lockp){
	int is_locked = -1, compare = -1;
	mpi_win_lock_t *lock = (mpi_win_lock_t *) lockp;
	lock->prev_owner_rank = -1;
	/* Get the last tail, if -1 replace with me */
	MPI_Compare_and_swap (&m_rank, &compare, &(lock->prev_owner_rank), MPI_INT,
			m_root, TAIL_DISP, m_mpi_win);
	MPI_Win_flush (m_root, m_mpi_win);
	/* Find if the last proc is holding lock */
	if (lock->prev_owner_rank != -1)
	{
	  MPI_Fetch_and_op (NULL, &is_locked, MPI_INT, lock->prev_owner_rank,
			LOCK_DISP, MPI_NO_OP, m_mpi_win);
	  MPI_Win_flush (lock->prev_owner_rank, m_mpi_win);

	  if (is_locked == 1)  // note here
		  return 0;
	}

	m_mpi_win_baseptr[LOCK_DISP] = 1;
	/* Add myself in tail */
	MPI_Fetch_and_op (&m_rank, &(lock->prev_owner_rank), MPI_INT, m_root,
			TAIL_DISP, MPI_REPLACE, m_mpi_win);
	MPI_Win_flush (m_root, m_mpi_win);
	/* Hold lock */
	//m_mpi_win_baseptr[LOCK_DISP] = 1;
	MPI_Win_sync (m_mpi_win);

	return 1;
}

}



