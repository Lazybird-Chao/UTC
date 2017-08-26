/*
 * internal_win.h
 *
 *  Created on: Jan 11, 2017
 *      Author: Chao
 */

#ifndef INCLUDE_SCOPEDDATA_INTERNAL_SHMEM_INTERNAL_WIN_H_
#define INCLUDE_SCOPEDDATA_INTERNAL_SHMEM_INTERNAL_WIN_H_

/*
 *
 */
#if ( defined(__GNUC__) && (__GNUC__ >= 3) ) || defined(__IBMC__) || defined(__INTEL_COMPILER) || defined(__clang__)
#  define unlikely(x_) __builtin_expect(!!(x_),0)
#  define likely(x_)   __builtin_expect(!!(x_),1)
#else
#  define unlikely(x_) (x_)
#  define likely(x_)   (x_)
#endif



/*
 *
 */
#define ONLY_MSPACES 1
#include "dlmalloc.h"

#include <mpi.h>

#include "mpi_win_lock.h"

/*
 * MPI_Accumulate functions will enable the ordering,
 * here we do not use it yet, just use MPI_put/get
 */

namespace iUtc{
class internal_MPIWin{
private:
	MPI_Comm *scoped_win_comm;
	MPI_Group scoped_win_group;

	int scoped_win_initialized;
	int scoped_win_finalized;
	int scoped_win_comm_size;
	int scoped_win_comm_rank;


	MPI_Win scoped_sheap_win;
	long    scoped_sheap_size;
	void *  scoped_sheap_base_ptr;

	mspace scoped_heap_mspace;

	enum window_id_e { _SHEAP_WINDOW = 0, _INVALID_WINDOW = -1 };
	enum coll_type_e { _BARRIER = 0, _BROADCAST = 1, _ALLREDUCE = 2, _FCOLLECT = 4, _COLLECT = 8};

	int scoped_win_root;
	MpiWinLock *scoped_win_lock;


public:
	internal_MPIWin(MPI_Comm *mpi_comm, long heap_size, int root);

	void scoped_win_init();
	void scoped_win_finalize();

	/* win_flush */
	void scoped_win_remote_sync();

	/* win sync, syn window private and public copy on the calling process */
	void scoped_win_local_sync();

	/* win_flush_local */
	void scoped_win_local_complete();
	void scoped_win_local_complete_pe(int pe);

	/* used internally only */
	void scoped_win_remote_sync_pe(int pe);

	/* return 0 on successful lookup, otherwise 1 */
	int scoped_win_offset(const void *address, const int pe,
	                         enum window_id_e * win_id, MPI_Aint * win_offset);

	void scoped_win_put(MPI_Datatype mpi_type, void *target, const void *source, size_t len, int pe);
	void scoped_win_get(MPI_Datatype mpi_type, void *target, const void *source, size_t len, int pe);

	void scoped_win_swap(MPI_Datatype mpi_type, void *output, void *remote, const void *input, int pe);
	void scoped_win_cswap(MPI_Datatype mpi_type, void *output, void *remote, const void *input, const void *compare, int pe);
	void scoped_win_add(MPI_Datatype mpi_type, void *remote, const void *input, int pe);
	void scoped_win_fadd(MPI_Datatype mpi_type, void *output, void *remote, const void *input, int pe);

	void scoped_win_wait(MPI_Datatype mpi_type, void *output, void *target, const void *value);
	void scoped_win_wait_util(MPI_Datatype mpi_type, void *output, void *target, int cond, const void *value);
	/*
	 *  All collective operations will be implemented in other place as
	 *  task utility functions; As those collective ops, such as bcast, gather,
	 *  reduce, allgather, allreduce, do not need one side communication patter,
	 *  they are just same as normal mpi-collective ops, so no need to implement
	 *  them here.
	 *  we can implement them to realize collective ops within a task that span
	 *  multiple nodes
	 */

	/*
	 *
	 */
	int get_scoped_win_comm_size();

	int get_scoped_win_comm_rank();

	long get_scoped_sheap_size();

	mspace get_heap_mspace();

	MpiWinLock *get_scoped_win_lock();

};

}


#endif /* INCLUDE_SCOPEDDATA_INTERNAL_SHMEM_INTERNAL_WIN_H_ */
