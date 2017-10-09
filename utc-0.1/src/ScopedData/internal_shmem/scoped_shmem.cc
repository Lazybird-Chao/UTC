/*
 * scoped_shmem.cc
 *
 */
#include "UtcBasics.h"
#include "scoped_shmem.h"
#include "internal_win.h"
#include "mpi_win_lock.h"
#include "dlmalloc.h"

#include <mpi.h>
#include <iostream>

//extern void* mspace_malloc(mspace msp, size_t bytes);
//extern void mspace_free(mspace msp, void* mem);
//extern void* mspace_realloc(mspace msp, void* mem, size_t newsize);
//extern void* mspace_memalign(mspace msp, size_t alignment, size_t bytes);

/*
 *
 */
using namespace iUtc;


void scoped_shmem_init_comm(internal_MPIWin &scoped_win){
	scoped_win.scoped_win_init();
	return;
}

void scoped_shmem_finalize_comm(internal_MPIWin &scoped_win){
	scoped_win.scoped_win_finalize();
}

int scoped_shmem_n_pes_comm(internal_MPIWin &scoped_win){
	return scoped_win.get_scoped_win_comm_size();
}

int scoped_shmem_my_pes_comm(internal_MPIWin &scoped_win){
	return scoped_win.get_scoped_win_comm_rank();
}

int scoped_shmem_pe_accessible(int pe,internal_MPIWin &scoped_win){
	return 0<=pe && pe<scoped_win.get_scoped_win_comm_size();
}


int scoped_shmem_addr_accessible(void* addr, int pe, internal_MPIWin &scoped_win){
	if(0<=pe && pe<scoped_win.get_scoped_win_comm_size()){
		enum window_id_e win_id;
		MPI_Aint win_offset;
		if(scoped_win.scoped_win_offset(addr, pe, &win_id, &win_offset) == 0)
			return 1;
	}
	return 0;
}

long scoped_shmem_addr_offset(void* addr, internal_MPIWin &scoped_win){
	enum window_id_e win_id;
	MPI_Aint win_offset;
	if(scoped_win.scoped_win_offset(addr, 0, &win_id, &win_offset) == 0)
		return (long)win_offset;
	else
		return -1;
}

/*
 *
 */
void *scoped_shmem_malloc(size_t size, internal_MPIWin &scoped_win){
	//std::cout<<scoped_win.get_heap_base_address()<<std::endl;
	//std::cout<<scoped_win.get_heap_mspace()<<std::endl;
	//mspace_malloc_stats(scoped_win.get_heap_mspace());
	void *mspace = scoped_win.get_heap_mspace();
	if (size > scoped_win.get_scoped_sheap_size() - mspace_footprint(mspace)){
		//std::cerr<<"Error: not enough shared space !!!"<<std::endl;
		return NULL;
	}
	void* m = mspace_malloc(mspace, size);
	//std::cout<<(long)m<<std::endl;

	/*
	 * int MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
               void *recvbuf, int recvcount, MPI_Datatype recvtype,
               int root, MPI_Comm comm)
	   int MPI_Bcast( void *buffer, int count, MPI_Datatype datatype, int root,
	   	   	   MPI_Comm comm )
	 */
#ifdef SHMEM_MALLOC_CHECK_SYMETRIC_ADDR
	enum window_id_e win_id;
	long offset;
	scoped_win.scoped_win_offset(m, 0, &win_id, &offset );
	long offset_compare = -1;
	if(scoped_win.get_scoped_win_comm_rank()==0)
		offset_compare = offset;
	MPI_Bcast(&offset_compare, 1, MPI_LONG, 0, *scoped_win.get_scoped_win_comm());
	if(offset_compare != offset){
		mspace_free(mspace, m);
		std::cerr<<"Error: var offsets in shared space are not same for all !!!"<<std::endl;
		return NULL;
	}
#endif

	return m;
}

void *scoped_shmem_align(size_t alignment, size_t size, internal_MPIWin &scoped_win){
	return mspace_memalign(scoped_win.get_heap_mspace(), alignment, size);
}

void *scoped_shmem_realloc(void* ptr, size_t size, internal_MPIWin &scoped_win){
	return mspace_realloc(scoped_win.get_heap_mspace(), ptr, size);
}

void scoped_shmem_free(void* ptr, internal_MPIWin &scoped_win){
	mspace_free(scoped_win.get_heap_mspace(), ptr);
	return;
}

/*
 *
 */
void scoped_shmem_float_p(float *addr, float value, int pe, internal_MPIWin &scoped_win){
	scoped_win.scoped_win_put(MPI_FLOAT, addr, &value, 1, pe);
}

void scoped_shmem_double_p(double *addr, double value, int pe,  internal_MPIWin &scoped_win){
	scoped_win.scoped_win_put(MPI_DOUBLE, addr, &value, 1, pe);
}

void scoped_shmem_char_p(char *addr, char value, int pe,  internal_MPIWin &scoped_win){
	scoped_win.scoped_win_put(MPI_CHAR, addr, &value, 1, pe);
}

void scoped_shmem_short_p(short *addr, short value, int pe,  internal_MPIWin &scoped_win){
	scoped_win.scoped_win_put(MPI_SHORT, addr, &value, 1, pe);
}

void scoped_shmem_int_p(int *addr, int value, int pe,  internal_MPIWin &scoped_win){
	scoped_win.scoped_win_put(MPI_INT, addr, &value, 1, pe);
}

void scoped_shmem_long_p(long *addr, long value, int pe,  internal_MPIWin &scoped_win){
	scoped_win.scoped_win_put(MPI_LONG, addr, &value, 1, pe);
}
/*
 *
 */
void scoped_shmem_float_put(float *target, const float *source, size_t len, int pe,  internal_MPIWin &scoped_win){
	scoped_win.scoped_win_put(MPI_FLOAT, target, source, len, pe);
}

void scoped_shmem_double_put(double *target, const double *source, size_t len, int pe,  internal_MPIWin &scoped_win){
	scoped_win.scoped_win_put(MPI_DOUBLE, target, source, len, pe);
}

void scoped_shmem_char_put(char *target, const char *source, size_t len, int pe,  internal_MPIWin &scoped_win){
	scoped_win.scoped_win_put(MPI_CHAR, target, source, len, pe);
}

void scoped_shmem_short_put(short *target, const short *source, size_t len, int pe,  internal_MPIWin &scoped_win){
	scoped_win.scoped_win_put(MPI_SHORT, target, source, len, pe);
}

void scoped_shmem_int_put(int *target, const int *source, size_t len, int pe,  internal_MPIWin &scoped_win){
	scoped_win.scoped_win_put(MPI_INT, target, source, len, pe);
}

void scoped_shmem_long_put(long *target, const long *source, size_t len, int pe,  internal_MPIWin &scoped_win){
	scoped_win.scoped_win_put(MPI_LONG, target, source, len, pe);
}

void scoped_shmem_put32(void *target, const void *source, size_t len, int pe,  internal_MPIWin &scoped_win){
	scoped_win.scoped_win_put(MPI_INT32_T, target, source, len, pe);
}

void scoped_shmem_put64(void *target, const void *source, size_t len, int pe,  internal_MPIWin &scoped_win){
	scoped_win.scoped_win_put(MPI_DOUBLE, target, source, len, pe);
}

void scoped_shmem_put128(void *target, const void *source, size_t len, int pe,  internal_MPIWin &scoped_win){
	scoped_win.scoped_win_put(MPI_C_DOUBLE_COMPLEX, target, source, len, pe);
}

void scoped_shmem_putmem(void *target, const void *source, size_t len, int pe,  internal_MPIWin &scoped_win){
	scoped_win.scoped_win_put(MPI_BYTE, target, source, len, pe);
}

/*void scoped_shmem_complexf_put(float complex * target, const float complex * source, size_t len, int pe,  internal_MPIWin &scoped_win){
	scoped_win.scoped_win_put(MPI_COMPLEX, target, source, len, pe);
}

void scoped_shmem_complexd_put(double complex * target, const double complex * source, size_t len, int pe,  internal_MPIWin &scoped_win){
	scoped_win.scoped_win_put(MPI_DOUBLE_COMPLEX, target, source, len, pe);
}*/

/*
 *
 */
float scoped_shmem_float_g(float *addr, int pe,  internal_MPIWin &scoped_win){
	float res;
	scoped_win.scoped_win_get(MPI_FLOAT, &res, addr, 1, pe);
	return res;
}

double scoped_shmem_double_g(double *addr, int pe,  internal_MPIWin &scoped_win){
	double res;
	scoped_win.scoped_win_get(MPI_DOUBLE, &res, addr, 1, pe);
	return res;
}

char scoped_shmem_char_g(char *addr, int pe,  internal_MPIWin &scoped_win){
	char res;
	scoped_win.scoped_win_get(MPI_CHAR, &res, addr, 1, pe);
	return res;
}

short scoped_shmem_short_g(short *addr, int pe,  internal_MPIWin &scoped_win){
	short res;
	scoped_win.scoped_win_get(MPI_SHORT, &res, addr, 1, pe);
	return res;
}

int scoped_shmem_int_g(int *addr, int pe,  internal_MPIWin &scoped_win){
	int res;
	scoped_win.scoped_win_get(MPI_INT, &res, addr, 1, pe);
	return res;
}

long scoped_shmem_long_g(long *addr, int pe,  internal_MPIWin &scoped_win){
	long res;
	scoped_win.scoped_win_get(MPI_LONG, &res, addr, 1, pe);
	return res;
}
/*
 *
 */
void scoped_shmem_float_get(float *target, const float *source, size_t len, int pe,  internal_MPIWin &scoped_win){
	scoped_win.scoped_win_get(MPI_FLOAT, target, source, len, pe);
}

void scoped_shmem_double_get(double *target, const double *source, size_t len, int pe,  internal_MPIWin &scoped_win){
	scoped_win.scoped_win_get(MPI_DOUBLE, target, source, len, pe);
}

void scoped_shmem_char_get(char *target, const char *source, size_t len, int pe,  internal_MPIWin &scoped_win){
	scoped_win.scoped_win_get(MPI_CHAR, target, source, len, pe);
}

void scoped_shmem_short_get(short *target, const short *source, size_t len, int pe,  internal_MPIWin &scoped_win){
	scoped_win.scoped_win_get(MPI_SHORT, target, source, len, pe);
}

void scoped_shmem_int_get(int *target, const int *source, size_t len, int pe,  internal_MPIWin &scoped_win){
	scoped_win.scoped_win_get(MPI_INT, target, source, len, pe);
}

void scoped_shmem_long_get(long *target, const long *source, size_t len, int pe,  internal_MPIWin &scoped_win){
	scoped_win.scoped_win_get(MPI_LONG, target, source, len, pe);
}

void scoped_shmem_get32(void *target, const void *source, size_t len, int pe,  internal_MPIWin &scoped_win){
	scoped_win.scoped_win_get(MPI_INT32_T, target, source, len, pe);
}

void scoped_shmem_get64(void *target, const void *source, size_t len, int pe,  internal_MPIWin &scoped_win){
	scoped_win.scoped_win_get(MPI_DOUBLE, target, source, len, pe);
}

void scoped_shmem_get128(void *target, const void *source, size_t len, int pe,  internal_MPIWin &scoped_win){
	scoped_win.scoped_win_get(MPI_C_DOUBLE_COMPLEX, target, source, len, pe);
}

void scoped_shmem_getmem(void *target, const void *source, size_t len, int pe,  internal_MPIWin &scoped_win){
	scoped_win.scoped_win_get(MPI_BYTE, target, source, len, pe);
}

/*void scoped_shmem_complexf_get(float complex * target, const float complex * source, size_t len, int pe,  internal_MPIWin &scoped_win){
	scoped_win.scoped_win_get(MPI_COMPLEX, target, source, len, pe);
}

void scoped_shmem_complexd_get(double complex * target, const double complex * source, size_t len, int pe,  internal_MPIWin &scoped_win){
	scoped_win.scoped_win_get(MPI_DOUBLE_COMPLEX, target, source, len, pe);
}*/

/*
 *
 */
float scoped_shmem_float_swap(float *target, float value, int pe,  internal_MPIWin &scoped_win){
	float res;
	scoped_win.scoped_win_swap(MPI_FLOAT, &res, target, &value, pe);
	return res;
}

double scoped_shmem_double_swap(double *target, double value, int pe,  internal_MPIWin &scoped_win){
	double res;
	scoped_win.scoped_win_swap(MPI_DOUBLE, &res, target, &value, pe);
	return res;
}

int scoped_shmem_int_swap(int *target, int value, int pe,  internal_MPIWin &scoped_win){
	int res;
	scoped_win.scoped_win_swap(MPI_INT, &res, target, &value, pe);
	return res;
}

long scoped_shmem_long_swap(long *target, long value, int pe,  internal_MPIWin &scoped_win){
	long res;
	scoped_win.scoped_win_swap(MPI_LONG, &res, target, &value, pe);
	return res;
}

long scoped_shmem_swap(long *target, long value, int pe,  internal_MPIWin &scoped_win){
	long res;
	scoped_win.scoped_win_swap(MPI_LONG, &res, target, &value, pe);
	return res;
}

/*
 *
 */
int scoped_shmem_int_cswap(int *target, int cond, int value, int pe,  internal_MPIWin &scoped_win){
	int res;
	scoped_win.scoped_win_cswap(MPI_INT, &res, target, &value, &cond, pe);
	return res;
}

long scoped_shmem_long_cswap(long *target, long cond, long value, int pe,  internal_MPIWin &scoped_win){
	long res;
	scoped_win.scoped_win_cswap(MPI_LONG, &res, target, &value, &cond, pe);
	return res;
}

/*
 *
 */
int scoped_shmem_int_fadd(int *target, int value, int pe,  internal_MPIWin &scoped_win){
	int res;
	scoped_win.scoped_win_fadd(MPI_INT, &res, target, &value, pe);
	return res;
}

long scoped_shmem_long_fadd(long *target, long value, int pe,  internal_MPIWin &scoped_win){
	long res;
	scoped_win.scoped_win_fadd(MPI_LONG, &res, target, &value, pe);
	return res;
}

int scoped_shmem_int_finc(int *target, int pe,  internal_MPIWin &scoped_win){
	int res;
	int value = 1;
	scoped_win.scoped_win_fadd(MPI_INT, &res, target, &value, pe);
	return res;
}

long scoped_shmem_long_finc(long *target, int pe,  internal_MPIWin &scoped_win){
	long res;
	long value = 1;
	scoped_win.scoped_win_fadd(MPI_INT, &res, target, &value, pe);
	return res;
}

void scoped_shmem_int_add(int *target, int value, int pe,  internal_MPIWin &scoped_win){
	scoped_win.scoped_win_add(MPI_INT, target, &value, pe);
}

void scoped_shmem_long_add(long *target, long value, int pe,  internal_MPIWin &scoped_win){
	scoped_win.scoped_win_add(MPI_INT, target, &value, pe);
}

void scoped_shmem_int_inc(int *target, int pe,  internal_MPIWin &scoped_win){
	int value = 1;
	scoped_win.scoped_win_add(MPI_INT, target, &value, pe);
}

void scoped_shmem_long_inc(long *target, int pe,  internal_MPIWin &scoped_win){
	long value = 1;
	scoped_win.scoped_win_add(MPI_INT, target, &value, pe);
}

/*
 *
 */
void scoped_shmem_short_wait(short *var, short value,  internal_MPIWin &scoped_win){
	short res;
	scoped_win.scoped_win_wait(MPI_SHORT, &res, var, &value);
}

void scoped_shmem_int_wait(int *var, int value,  internal_MPIWin &scoped_win){
	int res;
	scoped_win.scoped_win_wait(MPI_INT, &res, var, &value);
}

void scoped_shmem_long_wait(long *var, long value,  internal_MPIWin &scoped_win){
	long res;
	scoped_win.scoped_win_wait(MPI_LONG, &res, var, &value);

}

void scoped_shmem_wait(long *var, long value,  internal_MPIWin &scoped_win){
	long res;
	scoped_win.scoped_win_wait(MPI_LONG, &res, var, &value);
}

void scoped_shmem_short_wait_until(short *var, int cond, short value,  internal_MPIWin &scoped_win){
	short res;
	scoped_win.scoped_win_wait_until(MPI_SHORT, &res, var, cond, &value);
}

void scoped_shmem_int_wait_until(int *var, int cond, int value,  internal_MPIWin &scoped_win){
	int res;
	scoped_win.scoped_win_wait_until(MPI_INT, &res, var, cond, &value);
}

void scoped_shmem_long_wait_until(long *var, int cond, long value,  internal_MPIWin &scoped_win){
	long res;
	scoped_win.scoped_win_wait_until(MPI_LONG, &res, var, cond, &value);
}

void scoped_shmem_wait_until(long *var, int cond, long value,  internal_MPIWin &scoped_win){
	long res;
	scoped_win.scoped_win_wait_until(MPI_LONG, &res, var, cond, &value);
}


/*
 *
 */
void scoped_shmem_barrier( internal_MPIWin &scoped_win){
	scoped_win.scoped_win_remote_sync();
	scoped_win.scoped_win_local_sync();
	MPI_Barrier(*scoped_win.get_scoped_win_comm());
}

void scoped_shmem_quiet( internal_MPIWin &scoped_win){
	scoped_win.scoped_win_remote_sync();
	scoped_win.scoped_win_local_sync();
}

void scoped_shmem_quiet( internal_MPIWin &scoped_win, int pe){
	scoped_win.scoped_win_remote_sync_pe(pe);
	scoped_win.scoped_win_local_sync();
}

void scoped_shmem_fence( internal_MPIWin &scoped_win){
	/*
	 * TODO: use win_flush or win_flush_local ???
	 */
	//scoped_win.scoped_win_remote_sync();
	scoped_win.scoped_win_local_complete();

	scoped_win.scoped_win_local_sync();
}



void scoped_shmem_set_lock(long *lock,  internal_MPIWin &scoped_win){
	MpiWinLock *scoped_win_lock = scoped_win.get_scoped_win_lock();
	scoped_win_lock->lock(lock);

}

void scoped_shmem_clear_lock(long *lock,  internal_MPIWin &scoped_win){
	MpiWinLock *scoped_win_lock = scoped_win.get_scoped_win_lock();
	scoped_win_lock->unlock(lock);
}

int scoped_shmem_test_lock(long *lock,  internal_MPIWin &scoped_win){
	MpiWinLock *scoped_win_lock = scoped_win.get_scoped_win_lock();
	return scoped_win_lock->trylock(lock);
}








