/*
 * ishmem.h
 *
 *  Created on: Dec 6, 2016
 *      Author: chao
 */

#ifndef ISHMEM_H_
#define ISHMEM_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <complex.h>
#include <assert.h>

#include "mpi.h"
/*
 * ensure that it supports MPI-3 standard
 */

#define iSHMEM_CMP_EQ 1
#define iSHMEM_CMP_NE 2
#define iSHMEM_CMP_GT 3
#define iSHMEM_CMP_GE 4
#define iSHMEM_CMP_LT 5
#define iSHMEM_CMP_LE 6


void ishmem_init(void);
void ishmem_init_comm(MPI_Comm comm);
void ishmem_finalize(void);
void ishmem_finalize_comm(MPI_Comm comm);
void ishmem_exit(int status);
void ishmem_exit_comm(int status, MPI_Comm comm);


void ishmem_n_pes(void);
void ishmem_n_pes_comm(MPI_Comm comm);
void ishmem_my_pe(void);
void ishmem_my_pe_comm(MPI_Comm comm);


int ishmem_pe_accessible(int pe);
int ishmem_addr_accessible(void* addr, int pe);


void *shmal





#ifdef __cplusplus
}
#endif


#endif /* ISHMEM_H_ */
