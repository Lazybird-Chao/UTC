/*
 * mpi_win_lock.cc
 *
 *  Created on: Dec 12, 2016
 *      Author: chao
 */

#include "mpi_win_lock.h"

#define NEXT_DISP 1
#define PREV_DISP 0
#define TAIL_DISP 2 //Only has meaning in root
#define LOCK_DISP 3

namespace iUtc{

MpiWinLock::MpiWinLock(){
	m_root = 0;
	m_comm = &MPI_COMM_WORLD;
	MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);

	MPI_Info lock_info = MPI_INFO_NULL;
	MPI_Info_create(&lock_info);

	MPI_Info_set(lock_info,"same_size","true");


	MPI_Win_allocate(4*sizeof(int), sizeof(int), lock_info,
			*m_comm, &m_mpi_win_baseptr, &m_mpi_win);
	m_mpi_win_baseptr[NEXT_DISP] = -1;
	m_mpi_win_baseptr[PREV_DISP] = -1;
	m_mpi_win_baseptr[TAIL_DISP] = -1;
	m_mpi_win_baseptr[LOCK_DISP] = -1;
	MPI_Win_lock_all (m_root, m_mpi_win);

	MPI_Info_free(&lock_info);
}


MpiWinLock::MpiWinLock(MPI_Comm &comm, int root, int rank){
	m_root = root;
	m_comm = &comm;
	m_rank = rank;

	MPI_Info lock_info = MPI_INFO_NULL;
	MPI_Info_create(&lock_info);

	MPI_Info_set(lock_info,"same_size","true");


	MPI_Win_allocate(4*sizeof(int), sizeof(int), lock_info,
			*m_comm, &m_mpi_win_baseptr, &m_mpi_win);
	m_mpi_win_baseptr[NEXT_DISP] = -1;
	m_mpi_win_baseptr[PREV_DISP] = -1;
	m_mpi_win_baseptr[TAIL_DISP] = -1;
	m_mpi_win_baseptr[LOCK_DISP] = -1;
	MPI_Win_lock_all (m_root, m_mpi_win);

	MPI_Info_free(&lock_info);
}

MpiWinLock::~MpiWinLock(){
	MPI_Win_unlock_all(m_mpi_win);
	MPI_Win_free(&m_mpi_win);
}


void MpiWinLock::lock(long *lockp){
	MPI_Status status;
	mpi_win_lock_t *lock = (mpi_win_lock_t *) lockp;
	/* Replace myself with the last tail */
	MPI_Fetch_and_op (&m_rank, &(lock->prev), MPI_INT, m_root,
			TAIL_DISP, MPI_REPLACE, m_mpi_win);
	MPI_Win_flush (m_root, m_mpi_win);

	/* Previous proc holding lock will eventually notify */
	if (lock->prev != -1)
	{
	  /* Send my shmem_world_rank to previous proc's next */
	  MPI_Accumulate (&m_rank, 1, MPI_INT, lock->prev, NEXT_DISP,
			  1, MPI_INT, MPI_REPLACE, m_mpi_win);
	  MPI_Win_flush (lock->prev, m_mpi_win);
	  MPI_Probe (lock->prev, MPI_ANY_TAG, *m_comm, &status);
	}
	/* Hold lock */
	m_mpi_win_baseptr[LOCK_DISP] = 1;
	MPI_Win_sync (m_mpi_win);

	return;
}


void MpiWinLock::unlock(long *lockp){
	mpi_win_lock_t *lock = (mpi_win_lock_t *) lockp;
	/* Determine my next process */
	MPI_Fetch_and_op (NULL, &(lock->next), MPI_INT, m_rank,
			NEXT_DISP, MPI_NO_OP, m_mpi_win);
	MPI_Win_flush (m_rank, m_mpi_win);

	if (lock->next != -1)
	{
		MPI_Send (&m_rank, 1, MPI_INT, lock->next, -1, *m_comm);
	}
	/* Release lock */
	m_mpi_win_baseptr[LOCK_DISP] = -1;
	MPI_Win_sync (m_mpi_win);

	return;
}


int MpiWinLock::trylock(long *lockp){
	int is_locked = -1, nil = -1;
	mpi_win_lock_t *lock = (mpi_win_lock_t *) lockp;
	lock->prev = -1;
	/* Get the last tail, if -1 replace with me */
	MPI_Compare_and_swap (&m_rank, &nil, &(lock->prev), MPI_INT,
			m_root, TAIL_DISP, oshmpi_lock_win);
	MPI_Win_flush (m_root, m_mpi_win);
	/* Find if the last proc is holding lock */
	if (lock->prev != -1)
	{
	  MPI_Fetch_and_op (NULL, &is_locked, MPI_INT, lock->prev,
			LOCK_DISP, MPI_NO_OP, m_mpi_win);
	  MPI_Win_flush (lock->prev, m_mpi_win);

	  if (is_locked)
		  return 1;
	}
	/* Add myself in tail */
	MPI_Fetch_and_op (&m_rank, &(lock->prev), MPI_INT, m_root,
			TAIL_DISP, MPI_REPLACE, m_mpi_win);
	MPI_Win_flush (m_root, m_mpi_win);
	/* Hold lock */
	m_mpi_win_baseptr[LOCK_DISP] = 1;
	MPI_Win_sync (m_mpi_win);

	return 0;
}

}



