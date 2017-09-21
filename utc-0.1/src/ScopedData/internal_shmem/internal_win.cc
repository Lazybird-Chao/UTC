/*
 *
 */

#include "internal_win.h"
#include "dlmalloc.h"


#include <iostream>
#include <cstdlib>
#include <cstdio>

//extern "C"  void* create_mspace_with_base(void* base, size_t capacity, int locked);
//extern "C" size_t mspace_set_footprint_limit(mspace msp, size_t bytes);

namespace iUtc{
static inline int Type_size(MPI_Datatype mpi_type){
	int type_size;
	MPI_Type_size(mpi_type, &type_size);
	return type_size;
}

internal_MPIWin::internal_MPIWin(MPI_Comm *mpi_comm, long heap_size, int root){
	scoped_win_initialized = 0;
	scoped_win_finalized = 0;
	scoped_win_comm = mpi_comm;
	MPI_Comm_group(*scoped_win_comm, &scoped_win_group);
	MPI_Comm_rank(*scoped_win_comm, &scoped_win_comm_rank);
	MPI_Comm_size(*scoped_win_comm, &scoped_win_comm_size);

	scoped_sheap_size = heap_size;
	scoped_sheap_base_ptr = NULL;
	scoped_win_root = root;
	scoped_heap_mspace = NULL;


}

internal_MPIWin::~internal_MPIWin(){
	if(scoped_win_initialized && !scoped_win_finalized){
		scoped_win_finalize();
	}
	if(scoped_heap_mspace)
		destroy_mspace(scoped_heap_mspace);
	if(scoped_sheap_base_ptr)
		free(scoped_sheap_base_ptr);
}

void internal_MPIWin::scoped_win_init(){
	if(!scoped_win_initialized){
		MPI_Info sheap_info = MPI_INFO_NULL;
		MPI_Info_create(&sheap_info);
		MPI_Info_set(sheap_info, "same_size", "true");
		MPI_Info_set(sheap_info, "alloc_shm", "true");
		real_scoped_sheap_size = scoped_sheap_size + 128*sizeof(size_t); //be more safe to put here!!!
		/*int rc = MPI_Win_allocate((MPI_Aint)real_scoped_sheap_size,
									1, //displacement
									sheap_info,
		                            *scoped_win_comm,
									&scoped_sheap_base_ptr,
									&scoped_sheap_win);*/
		scoped_sheap_base_ptr = malloc(real_scoped_sheap_size);
		int rc = MPI_Win_create(scoped_sheap_base_ptr,
								real_scoped_sheap_size,
								1,
								sheap_info,
								*scoped_win_comm,
								&scoped_sheap_win);

		if (rc!=MPI_SUCCESS) {
			char errmsg[MPI_MAX_ERROR_STRING];
			int errlen;
			MPI_Error_string(rc, errmsg, &errlen);
			printf("MPI_Win_allocate_shared error message = %s\n",errmsg);
			//MPI_Abort(*scoped_win_comm, rc);
		}

		 MPI_Win_lock_all(MPI_MODE_NOCHECK /* use 0 instead if things break */,
				 scoped_sheap_win);
		 /* dlmalloc mspace constructor.
		 * locked may not need to be 0 if SHMEM makes no multithreaded access... */
		 /* Part (less than 128*sizeof(size_t) bytes) of this space is used for bookkeeping,
		 * so the capacity must be at least this large */
		 //scoped_sheap_size += 128*sizeof(size_t);
		 scoped_heap_mspace = create_mspace_with_base(scoped_sheap_base_ptr,
				 	 	 	 	 	 	 	 real_scoped_sheap_size,
											 0 /* locked */);
		 /*
		  * should set the limit size of mspace;
		  * similar to shmem_symetirc space size, that shared data can not
		  * exceed this size from shmalloc()
		  *
		  * we have do proximate check in internal_shmem_malloc()
		  *
		  */
		 //mspace_set_footprint_limit(scoped_heap_mspace, real_scoped_sheap_size);

		 MPI_Info_free(&sheap_info);

		 scoped_win_lock = new MpiWinLock(scoped_win_comm, scoped_win_root, scoped_win_comm_rank );

		 MPI_Barrier(*scoped_win_comm);
	}

	scoped_win_initialized = 1;
	return;
}

void internal_MPIWin::scoped_win_finalize(){
	if (scoped_win_initialized && !scoped_win_finalized){
		if(scoped_win_lock){
			delete scoped_win_lock;
		}

		// do we need to destroy the mspace???
		//destroy_mspace(scoped_heap_mspace);

		MPI_Barrier(*scoped_win_comm);
		MPI_Win_unlock_all(scoped_sheap_win);
		MPI_Win_free(&scoped_sheap_win);

		scoped_win_finalized = 1;
	}
	return;
}

void internal_MPIWin::scoped_win_remote_sync(){
	MPI_Win_flush_all(scoped_sheap_win);
}

void internal_MPIWin::scoped_win_remote_sync_pe(int pe){
	MPI_Win_flush(pe, scoped_sheap_win);
}

void internal_MPIWin::scoped_win_local_sync(){
	MPI_Win_sync(scoped_sheap_win);
}

void internal_MPIWin::scoped_win_local_complete(){
	MPI_Win_flush_local_all(scoped_sheap_win);
}

void internal_MPIWin::scoped_win_local_complete_pe(int pe){
	MPI_Win_flush_local(pe, scoped_sheap_win);
}

/* return 0 on successful lookup, otherwise 1 */
int internal_MPIWin::scoped_win_offset(const void *address, const int pe, /* IN  */
        enum window_id_e * win_id,   /* OUT */
        MPI_Aint * win_offset)       /* OUT */ {
	/*
	 * notice the long here may cause problem, it only suits 64bit system
	 */
	long sheap_offset = (long)address - (long)scoped_sheap_base_ptr;
	if (0 <= sheap_offset && sheap_offset <= scoped_sheap_size) {
		*win_offset = sheap_offset;
		*win_id     = _SHEAP_WINDOW;
		return 0;
	}
	else{
		*win_offset = (MPI_Aint)NULL;
		*win_id      = _INVALID_WINDOW;
		return 1;
	}
}


void internal_MPIWin::scoped_win_put(MPI_Datatype mpi_type,
		void *target,
		const void *source,
		size_t len,
		int pe){
	enum window_id_e win_id;
	MPI_Aint win_offset;

	if(scoped_win_offset(target, pe, &win_id, &win_offset)){
		/*
		 * error
		 */
		printf("Error, scoped_win_put offset failed");
		exit(1);
	}

	/*
	 *  do not use MPIX yet. For big mpi feature, where element count would be
	 *  large than MAX_INT
	 */

	/*
	 *  do not use mpi ordering yet, which use MPI_Accumulate() for put
	 */

	MPI_Put(source, len, mpi_type,                   /* origin */
	              pe, (MPI_Aint)win_offset, len, mpi_type, /* target */
	                scoped_sheap_win);
	MPI_Win_flush_local(pe, scoped_sheap_win);

	return;

}


void internal_MPIWin::scoped_win_get(MPI_Datatype mpi_type,
		void *target,
		const void *source,
		size_t len,
		int pe){
	enum window_id_e win_id;
	MPI_Aint win_offset;

	if(scoped_win_offset(source, pe, &win_id, &win_offset)){
		/*
		 * error
		 */
		printf("Error, scoped_win_put offset failed");
		exit(1);
	}

	/*
	 *  do not use MPIX yet. For big mpi feature, where element count would be
	 *  large than MAX_INT
	 */

	/*
	 *  do not use mpi ordering yet, which use MPI_Get_Accumulate() for get
	 */
	MPI_Get(target, len, mpi_type,                   /* result */
			pe, (MPI_Aint)win_offset, len, mpi_type, /* remote */
			scoped_sheap_win);
	MPI_Win_flush_local(pe, scoped_sheap_win);
}


void internal_MPIWin::scoped_win_swap(MPI_Datatype mpi_type,
		void *output,
		void *remote,
		const void *input,
		int pe){
	enum window_id_e win_id;
	MPI_Aint win_offset;

	if(scoped_win_offset(remote, pe, &win_id, &win_offset)){
		/*
		 * error
		 */
		printf("Error, scoped_win_put offset failed");
		exit(1);
	}

	MPI_Fetch_and_op(input, output, mpi_type, pe, win_offset, MPI_REPLACE, scoped_sheap_win);
	MPI_Win_flush(pe, scoped_sheap_win);

	return;
}

void internal_MPIWin::scoped_win_cswap(MPI_Datatype mpi_type,
		void *output,
		void *remote,
		const void *input,
		const void *compare,
		int pe){
	enum window_id_e win_id;
	MPI_Aint win_offset;

	if(scoped_win_offset(remote, pe, &win_id, &win_offset)){
		/*
		 * error
		 */
		printf("Error, scoped_win_put offset failed");
		exit(1);
	}

	MPI_Compare_and_swap(input, compare, output, mpi_type, pe, win_offset, scoped_sheap_win);
	MPI_Win_flush(pe, scoped_sheap_win);

	return;
}


void internal_MPIWin::scoped_win_add(MPI_Datatype mpi_type,
		void *remote,
		const void *input,
		int pe){
	enum window_id_e win_id;
	MPI_Aint win_offset;

	if(scoped_win_offset(remote, pe, &win_id, &win_offset)){
		/*
		 * error
		 */
		printf("Error, scoped_win_put offset failed");
		exit(1);
	}

	MPI_Accumulate(input, 1, mpi_type, pe, win_offset, 1, mpi_type, MPI_SUM, scoped_sheap_win);
	MPI_Win_flush_local(pe, scoped_sheap_win);

	return;
}


void internal_MPIWin::scoped_win_fadd(MPI_Datatype mpi_type,
		void *output,
		void *remote,
		const void *input,
		int pe){
	enum window_id_e win_id;
	MPI_Aint win_offset;

	if(scoped_win_offset(remote, pe, &win_id, &win_offset)){
		/*
		 * error
		 */
		printf("Error, scoped_win_put offset failed");
		exit(1);
	}

	MPI_Fetch_and_op(input, output, mpi_type, pe, win_offset, MPI_SUM, scoped_sheap_win);
	MPI_Win_flush(pe, scoped_sheap_win);

	return;
}

void internal_MPIWin::scoped_win_wait(MPI_Datatype mpi_type,
		void *output,
		void *target,
		const void *value){
	enum window_id_e win_id;
	MPI_Aint win_offset;
	if(scoped_win_offset(target, scoped_win_comm_rank, &win_id, &win_offset)){
		/*
		 * error
		 */
		printf("Error, scoped_win_put offset failed");
		exit(1);
	}
	if(Type_size(mpi_type)==8){
		long *o = (long*)output;
		long *v = (long*)value;
		*o = *v;
		while(*o == *v){
			MPI_Fetch_and_op(NULL, output, mpi_type, scoped_win_comm_rank,
					win_offset, MPI_NO_OP, scoped_sheap_win);
			MPI_Win_flush_local(scoped_win_comm_rank, scoped_sheap_win);
		}
	}
	else{
		std::cerr<<"no implemented type in scoped_win_wait!!!"<<std::endl;
	}
}

void internal_MPIWin::scoped_win_wait_until(MPI_Datatype mpi_type,
		void *output,
		void *target,
		int cond,
		const void *value){
	enum window_id_e win_id;
	MPI_Aint win_offset;
	if(scoped_win_offset(target, scoped_win_comm_rank, &win_id, &win_offset)){
		/*
		 * error
		 */
		printf("Error, scoped_win_put offset failed");
		exit(1);
	}

	bool comp_res =false;
	if(Type_size(mpi_type)==8){
		long *o = (long*)output;
		long *v = (long*)value;
	while(!comp_res){
		MPI_Fetch_and_op(NULL, output, mpi_type, scoped_win_comm_rank,
					win_offset, MPI_NO_OP, scoped_sheap_win);
		MPI_Win_flush_local(scoped_win_comm_rank, scoped_sheap_win);
		switch(cond){
		case 1:
			if(*o == *v)
				comp_res = true;
			break;
		case 2:
			if(*o != *v)
				comp_res = true;
			break;
		case 3:
			if(*o > *v)
				comp_res = true;
			break;
		case 4:
			if(*o >= *v)
				comp_res = true;
			break;
		case 5:
			if(*o < *v)
				comp_res = true;
			break;
		case 6:
			if(*o <= *v)
				comp_res = true;
			break;
		default:
			std::cerr<<"invalid comparison in scoped_win_wait!!!"<<std::endl;

		}
	}
	}
	else{
		std::cerr<<"no implemented type in scoped_win_wait!!!"<<std::endl;
	}
}

int internal_MPIWin::get_scoped_win_comm_size(){
	return scoped_win_comm_size;
}

int internal_MPIWin::get_scoped_win_comm_rank(){
	return scoped_win_comm_rank;
}

long internal_MPIWin::get_scoped_sheap_size(){
	return scoped_sheap_size;
}

void* internal_MPIWin::get_heap_mspace(){
	return scoped_heap_mspace;
}

void* internal_MPIWin::get_heap_base_address(){
	return scoped_sheap_base_ptr;
}

MpiWinLock *internal_MPIWin::get_scoped_win_lock(){
	return scoped_win_lock;
}

MPI_Comm *internal_MPIWin::get_scoped_win_comm(){
	return scoped_win_comm;
}

}// end namespace iUtc
