/*
 * commonUtil.h
 *
 *  Created on: Nov 19, 2016
 *      Author: chao
 */

#ifndef UTC_COMMONUTIL_H_
#define UTC_COMMONUTIL_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <pthread.h>
#include <linux/futex.h>
#include <stdint.h>
#include <limits.h>
#include <err.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef _GNU_SOURCE
	#define _GNU_SOURCE
#endif
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/time.h>

/* Compile Barrier */
#define mem_barrier() asm volatile("": : :"memory")
#define cpu_relax()  __asm__ __volatile__ ( "pause" : : : )

/* Atomic add, returning the new value after the addition */
#define atomic_add(P, V) __sync_add_and_fetch((P), (V))

/* Atomic add, returning the value before the addition */
#define atomic_xadd(P, V) __sync_fetch_and_add((P), (V))

/* Atomic or */
#define atomic_or(P, V) __sync_or_and_fetch((P), (V))

/* Force a read of the variable */
//#define atomic_read(V) (*(volatile typeof(V) *)&(V))

/* Atomic set and clear a bit, given by position */
#define atomic_set_bit(P, V) __sync_or_and_fetch((P), 1<<(V))
#define atomic_clear_bit(P, V) __sync_and_and_fetch((P), ~(1<<(V)))

/* Compare Exchange */
#define cmpxchg(P, O, N) __sync_val_compare_and_swap((P), (O), (N))



/* Atomic 8 bit exchange */
static inline unsigned xchg_8(void *ptr, unsigned char x)
{
	__asm__ __volatile__("xchgb %0,%1"
				:"=r" ((unsigned char) x)
				:"m" (*(volatile unsigned char *)ptr), "0" (x)
				:"memory");

	return x;
}

/* Atomic 32 bit exchange */
static inline unsigned xchg_32(void *ptr, unsigned x)
{
	__asm__ __volatile__("xchgl %0,%1"
				:"=r" ((unsigned) x)
				:"m" (*(volatile unsigned *)ptr), "0" (x)
				:"memory");

	return x;
}

/* Atomic 64 bit exchange */
static inline unsigned long long xchg_64(void *ptr, unsigned long long x)
{
	__asm__ __volatile__("xchgq %0,%1"
				:"=r" ((unsigned long long) x)
				:"m" (*(volatile unsigned long long *)ptr), "0" (x)
				:"memory");

	return x;
}


static long sys_futex(void *addr1, int op, int val1, struct timespec *timeout, void *addr2, int val3)
{
	return syscall(SYS_futex, addr1, op, val1, timeout, addr2, val3);
}


#ifdef __cplusplus
}
#endif


#endif /* COMMONUTIL_H_ */
