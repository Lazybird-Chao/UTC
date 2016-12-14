/*
 * mpi_win_lock.h
 *
 *  Created on: Dec 12, 2016
 *      Author: chao
 */

#ifndef MPI_WIN_LOCK_H_
#define MPI_WIN_LOCK_H_

#include <mpi.h>

typedef struct{
	int prev_owner_rank;
	int next_owner_rank;
} mpi_win_lock_t;

namespace iUtc{
class MpiWinLock{
private:
	MPI_Win m_mpi_win;
	int *m_mpi_win_baseptr;

	MPI_Comm *m_comm;
	int m_root;
	int m_rank;
public:
	MpiWinLock();

	MpiWinLock(MPI_Comm &comm, int root);

	~MpiWinLock();

	void lock(long *lockp);
	void unlock(long *lockp);
	int trylock(long *lockp);
};

}


#endif /* MPI_WIN_LOCK_H_ */
