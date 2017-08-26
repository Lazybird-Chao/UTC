/*
 * scoped_shmem.h
 *
 *  Created on: Dec 6, 2016
 *      Author: chao
 */

#ifndef SCOPED_SHMEM_H_
#define SCOPED_SHMEM_H_

#include "internal_win.h"


/*
#ifdef __cplusplus
extern "C" {
#endif
*/

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cassert>

#include <mpi.h>
/*
 * ensure that it supports MPI-3 standard
 */


/*
 * with this define, we will check if allocated address has same offset
 * in different process, may slow down the shared var allocation procedure
 */
#define SHMEM_MALLOC_CHECK_SYMETRIC_ADDR

#define SCOPED_SHMEM_CMP_EQ 1
#define SCOPED_SHMEM_CMP_NE 2
#define SCOPED_SHMEM_CMP_GT 3
#define SCOPED_SHMEM_CMP_GE 4
#define SCOPED_SHMEM_CMP_LT 5
#define SCOPED_SHMEM_CMP_LE 6

using namespace iUtc;

//void scoped_shmem_init(void);
void scoped_shmem_init_comm( internal_MPIWin &scoped_win);
//void scoped_shmem_finalize(void);
void scoped_shmem_finalize_comm( internal_MPIWin &scoped_win);
//void scoped_shmem_exit(int status);



//int scoped_shmem_n_pes(void);
int scoped_shmem_n_pes_comm( internal_MPIWin &scoped_win);
//int scoped_shmem_my_pe(void);
int scoped_shmem_my_pe_comm( internal_MPIWin &scoped_win);


int scoped_shmem_pe_accessible(int pe, internal_MPIWin &scoped_win);
int scoped_shmem_addr_accessible(void* addr, int pe,  internal_MPIWin &scoped_win);
long scoped_shmem_addr_offset(void* addr, internal_MPIWin &scoped_win);

void *scoped_shmem_malloc(size_t size,  internal_MPIWin &scoped_win);
void *scoped_shmem_align(size_t alignment, size_t size,  internal_MPIWin &scoped_win);
void *scoped_shmem_realloc(void* ptr, size_t size,  internal_MPIWin &scoped_win);
void scoped_shmem_free(void* ptr,  internal_MPIWin &scoped_win);


//void *scoped_shmem_ptr(void *target, int pe);

/*
 * single element store
 */
void scoped_shmem_float_p(float *addr, float value, int pe,  internal_MPIWin &scoped_win);
void scoped_shmem_double_p(double *addr, double value, int pe,  internal_MPIWin &scoped_win);
void scoped_shmem_char_p(char *addr, char value, int pe,  internal_MPIWin &scoped_win);
void scoped_shmem_short_p(short *addr, short value, int pe,  internal_MPIWin &scoped_win);
void scoped_shmem_int_p(int *addr, int value, int pe,  internal_MPIWin &scoped_win);
void scoped_shmem_long_p(long *addr, long value, int pe,  internal_MPIWin &scoped_win);
/*
 * block store
 */
void scoped_shmem_float_put(float *target, const float *source, size_t len, int pe,  internal_MPIWin &scoped_win);
void scoped_shmem_double_put(double *target, const double *source, size_t len, int pe,  internal_MPIWin &scoped_win);
void scoped_shmem_char_put(char *target, const char *source, size_t nelems, int pe,  internal_MPIWin &scoped_win);
void scoped_shmem_short_put(short *target, const short *source, size_t len, int pe,  internal_MPIWin &scoped_win);
void scoped_shmem_int_put(int *target, const int *source, size_t len, int pe,  internal_MPIWin &scoped_win);
void scoped_shmem_long_put(long *target, const long *source, size_t len, int pe,  internal_MPIWin &scoped_win);
void scoped_shmem_put32(void *target, const void *source, size_t len, int pe,  internal_MPIWin &scoped_win);
void scoped_shmem_put64(void *target, const void *source, size_t len, int pe,  internal_MPIWin &scoped_win);
void scoped_shmem_put128(void *target, const void *source, size_t len, int pe,  internal_MPIWin &scoped_win);
void scoped_shmem_putmem(void *target, const void *source, size_t len, int pe,  internal_MPIWin &scoped_win);
//void scoped_shmem_complexf_put(float complex * target, const float complex * source, size_t nelems, int pe,  internal_MPIWin &scoped_win);
//void scoped_shmem_complexd_put(double complex * target, const double complex * source, size_t nelems, int pe,  internal_MPIWin &scoped_win);

/*
 * single element load
 */
float scoped_shmem_float_g(float *addr, int pe,  internal_MPIWin &scoped_win);
double scoped_shmem_double_g(double *addr, int pe,  internal_MPIWin &scoped_win);
char scoped_shmem_char_g(char *addr, int pe,  internal_MPIWin &scoped_win);
short scoped_shmem_short_g(short *addr, int pe,  internal_MPIWin &scoped_win);
int scoped_shmem_int_g(int *addr, int pe,  internal_MPIWin &scoped_win);
long scoped_shmem_long_g(long *addr, int pe,  internal_MPIWin &scoped_win);
/*
 * block load
 */
void scoped_shmem_float_get(float *target, const float *source, size_t len, int pe,  internal_MPIWin &scoped_win);
void scoped_shmem_double_get(double *target, const double *source, size_t len, int pe,  internal_MPIWin &scoped_win);
void scoped_shmem_char_get(char *target, const char *source, size_t len, int pe,  internal_MPIWin &scoped_win);
void scoped_shmem_short_get(short *target, const short *source, size_t len, int pe,  internal_MPIWin &scoped_win);
void scoped_shmem_int_get(int *target, const int *source, size_t len, int pe,  internal_MPIWin &scoped_win);
void scoped_shmem_long_get(long *target, const long *source, size_t len, int pe,  internal_MPIWin &scoped_win);
void scoped_shmem_get32(void *target, const void *source, size_t len, int pe,  internal_MPIWin &scoped_win);
void scoped_shmem_get64(void *target, const void *source, size_t len, int pe,  internal_MPIWin &scoped_win);
void scoped_shmem_get128(void *target, const void *source, size_t len, int pe,  internal_MPIWin &scoped_win);
void scoped_shmem_getmem(void *target, const void *source, size_t len, int pe,  internal_MPIWin &scoped_win);
//void scoped_shmem_complexf_get(float complex * target, const float complex * source, size_t nelems, int pe,  internal_MPIWin &scoped_win);
//void scoped_shmem_complexd_get(double complex * target, const double complex * source, size_t nelems, int pe,  internal_MPIWin &scoped_win);


/*
 * TODO: stride array put/get not implemente yet
 */


/* atomic fetch and swap */
float scoped_shmem_float_swap(float *target, float value, int pe,  internal_MPIWin &scoped_win);
double scoped_shmem_double_swap(double *target, double value, int pe,  internal_MPIWin &scoped_win);
int scoped_shmem_int_swap(int *target, int value, int pe,  internal_MPIWin &scoped_win);
long scoped_shmem_long_swap(long *target, long value, int pe,  internal_MPIWin &scoped_win);
long scoped_shmem_swap(long *target, long value, int pe,  internal_MPIWin &scoped_win);

/* atomic fetch and condition swap */
int scoped_shmem_int_cswap(int *target, int cond, int value, int pe,  internal_MPIWin &scoped_win);
long scoped_shmem_long_cswap(long *target, long cond, long value, int pe,  internal_MPIWin &scoped_win);
//long long scoped_shmem_longlong_cswap(long long * target, long long cond, long long value, int pe, const internal_MPIWin &scoped_win);

/* Atomic Memory fetch-and-operate Routines -- Fetch and Add */
int scoped_shmem_int_fadd(int *target, int value, int pe,  internal_MPIWin &scoped_win);
long scoped_shmem_long_fadd(long *target, long value, int pe,  internal_MPIWin &scoped_win);
//long long scoped_shmem_longlong_fadd(long long *target, long long value, int pe);

/* Atomic Memory fetch-and-operate Routines -- Fetch and Increment */
int scoped_shmem_int_finc(int *target, int pe,  internal_MPIWin &scoped_win);
long scoped_shmem_long_finc(long *target, int pe,  internal_MPIWin &scoped_win);
//long long scoped_shmem_longlong_finc(long long *target, int pe);

/*  Atomic Memory Operation Routines -- Add */
void scoped_shmem_int_add(int *target, int value, int pe,  internal_MPIWin &scoped_win);
void scoped_shmem_long_add(long *target, long value, int pe,  internal_MPIWin &scoped_win);
//void scoped_shmem_longlong_add(long long *target, long long value, int pe);

/* Atomic Memory Operation Routines -- Increment */
void scoped_shmem_int_inc(int *target, int pe,  internal_MPIWin &scoped_win);
void scoped_shmem_long_inc(long *target, int pe,  internal_MPIWin &scoped_win);
//void scoped_shmem_longlong_inc(long long *target, int pe);

/* Point-to-Point Synchronization Routines -- Wait*/
void scoped_shmem_short_wait(short *var, short value,  internal_MPIWin &scoped_win);
void scoped_shmem_int_wait(int *var, int value,  internal_MPIWin &scoped_win);
void scoped_shmem_long_wait(long *var, long value,  internal_MPIWin &scoped_win);
//void scoped_shmem_longlong_wait(long long *var, long long value);
void scoped_shmem_wait(long *var, long cmp_value,  internal_MPIWin &scoped_win);

/* Point-to-Point Synchronization Routines -- Wait Until*/
void scoped_shmem_short_wait_until(short *var, int cond, short value,  internal_MPIWin &scoped_win);
void scoped_shmem_int_wait_until(int *var, int cond, int value,  internal_MPIWin &scoped_win);
void scoped_shmem_long_wait_until(long *var, int cond, long value,  internal_MPIWin &scoped_win);
//void scoped_shmem_longlong_wait_until(long long *var, int cond,long long value);
void scoped_shmem_wait_until(long *var, int cmp, long value,  internal_MPIWin &scoped_win);



/* Barrier Synchronization Routines */
//void scoped_shmem_barrier(void);
void scoped_shmem_barrier( internal_MPIWin &scoped_win);

void scoped_shmem_quiet( internal_MPIWin &scoped_win);
void scoped_shmem_fence( internal_MPIWin &scoped_win);

/*
 *
 * All the with-in task collective operations will be implemented separately;
 * They are not bound to one-side shared data access.
 *
 *
// Reduction operation
void scoped_shmem_float_op_to_all(float *target, float *source, int nreduce,
                            int PE_start, int logPE_stride, int PE_size,
                            float *pWrk, long *pSync);
void scoped_shmem_double_op_to_all(double *target, double *source, int nreduce,
                             int PE_start, int logPE_stride, int PE_size,
                             double *pWrk, long *pSync);
void scoped_shmem_short_op_to_all(short *target, short *source, int nreduce,
                            int PE_start, int logPE_stride, int PE_size,
                            short *pWrk, long *pSync);
void scoped_shmem_int_op_to_all(int *target, int *source, int nreduce,
                          int PE_start, int logPE_stride, int PE_size,
                          int *pWrk, long *pSync);
void scoped_shmem_long_op_to_all(long *target, long *source, int nreduce,
                           int PE_start, int logPE_stride, int PE_size,
                           long *pWrk, long *pSync);
void scoped_shmem_longlong_op_to_all(long long *target, long long *source,
                               int nreduce, int PE_start, int logPE_stride,
                               int PE_size, long long *pWrk, long *pSync);
void scoped_shmem_complexf_op_to_all(float complex *target, float complex *source,
                               int nreduce, int PE_start, int logPE_stride,
                               int PE_size, float complex *pWrk, long *pSync);
void scoped_shmem_complexd_op_to_all(double complex *target, double complex *source,
                               int nreduce, int PE_start, int logPE_stride,
                               int PE_size, double complex *pWrk, long *pSync);


// Gather operation
void scoped_shmem_collect32(void *target, const void *source, size_t nlong,
                     int PE_start, int logPE_stride, int PE_size, long *pSync);
void scoped_shmem_collect64(void *target, const void *source, size_t nlong,
                     int PE_start, int logPE_stride, int PE_size, long *pSync);


// Broadcast operation
void scoped_shmem_broadcast32(void *target, const void *source, size_t nlong,
                       int PE_root, int PE_start, int logPE_stride,
                       int PE_size, long *pSync);
void scoped_shmem_broadcast64(void *target, const void *source, size_t nlong,
                       int PE_root, int PE_start, int logPE_stride,
                       int PE_size, long *pSync);
*/



/*
 * Lock operation
 * There may be problem with the underlying implementation
 */
void scoped_shmem_set_lock(long *lock,  internal_MPIWin &scoped_win);
void scoped_shmem_clear_lock(long *lock,  internal_MPIWin &scoped_win);
int scoped_shmem_test_lock(long *lock,  internal_MPIWin &scoped_win);


/*
#ifdef __cplusplus
}
#endif
*/


#endif /* SCOPED_SHMEM_H_ */
