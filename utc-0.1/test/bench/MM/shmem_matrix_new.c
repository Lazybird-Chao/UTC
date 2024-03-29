/*
 *
 * Copyright (c) 2011 - 2015
 *   University of Houston System and Oak Ridge National Laboratory.
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * o Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 *
 * o Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in the
 *   documentation and/or other materials provided with the distribution.
 *
 * o Neither the name of the University of Houston System, Oak Ridge
 *   National Laboratory nor the names of its contributors may be used to
 *   endorse or promote products derived from this software without specific
 *   prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 * TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Program to calculate the product of 2 matrices A and B based on block
 * distribution. Adopted from the mpi implementation of matrix muliplication
 * based on 1D block-column distribution.
 *
 *
 * 		Use shmem_get/put to get the whole submatrix at a time, not one row
 *
 *
 * */

#include <stdio.h>
#include <stdlib.h>
#include <shmem.h>
#include <sys/time.h>
#include <unistd.h>
#include "../helper_printtime.h"

//#define DEBUG 1

double
gettime ()
{
    struct timeval tv;
    gettimeofday (&tv, 0);
    return (tv.tv_sec * 1000000 + tv.tv_usec);
}

double
dt (double *tv1, double *tv2)
{
    return (*tv1 - *tv2);
}


/* set the number of rows and column here */
//#define ROWS 1024
//#define COLUMNS 1024
int ROWS=1024;
int COLUMNS=1024;

// routine to print the partial array
void
print_array (double **array, int blocksize)
{
    int i, j;
    for (i = 0; i < ROWS; i++) {
        for (j = 0; j < blocksize; j++) {
            printf ("%f  ", array[i][j]);
        }                       // end for loop j
        printf ("\n");
    }                           // end for loop i
    printf ("\n");
    printf ("\n");
}


// needed for reduction operation
long pSync[_SHMEM_REDUCE_SYNC_SIZE];
double pWrk[_SHMEM_REDUCE_MIN_WRKDATA_SIZE];

// global shmem_accesible
double maxtime;
double t, tv[2];
double t2, tv2[2];
double maxtime2;
int
main (int argc, char **argv)
{
	shmem_init();
    int i, j, k;
    int blocksize;
    int rank, size, nextpe;
    int p, np;                  // round and number of process
    double **a_local, **b_local;
    double **c_local;
    int B_matrix_displacement;
    for (i = 0; i < _SHMEM_REDUCE_SYNC_SIZE; i += 1)
        pSync[i] = _SHMEM_SYNC_VALUE;

    //tv[0] = gettime ();
    //shmem_init ();
    rank = shmem_my_pe ();
    size = shmem_n_pes ();
    np = size;                  // number of processes
    
    if(argc == 2 ){
    	COLUMNS = ROWS = atoi(argv[1]);
    }
    //printf("on rank %ld : %ld\n", _SHMEM_REDUCE_SYNC_SIZE, _SHMEM_SYNC_VALUE);

    blocksize = COLUMNS / np;   // block size
    B_matrix_displacement = rank * blocksize;

    // initialize the input arrays
    shmem_barrier_all ();
    //printf ("first barrier %d\n", rank);
    a_local = (double **) shmem_malloc (ROWS * sizeof (double *));
    a_local[0]= (double *) shmem_malloc (ROWS * blocksize * sizeof (double));
    b_local = (double **) shmem_malloc (ROWS * sizeof (double *));
    b_local[0] = (double *) shmem_malloc (ROWS * blocksize * sizeof (double));
    c_local = (double **) shmem_malloc (ROWS * sizeof (double *));
    c_local[0] = (double *) shmem_malloc (ROWS * blocksize * sizeof (double));
    
    for (i = 1; i < ROWS; i++) {
        a_local[i] = a_local[i-1] + blocksize;
        b_local[i] = b_local[i-1] + blocksize;
        c_local[i] = c_local[i-1] + blocksize;
        //printf ("matrix a from %d %d\n", rank, i);
    }
    for(i=0; i< ROWS; i++){
        for (j = 0; j < blocksize; j++) {
            a_local[i][j] = i + 1 * j + 1 * rank + 1;   // random values
            b_local[i][j] = i + 2 * j + 2 * rank + 1;   // random values
            c_local[i][j] = 0.0;
        }
    }
    //printf ("before second barrier %d\n", rank);
    shmem_barrier_all ();
    //printf ("after second barrier %d\n", rank);
#ifdef DEBUG                    // print the input arrays from root process if
                                // DEBUG enabled
    //if (rank == 0) {
        printf ("matrix a from %d %d\n", rank, blocksize);
        //print_array (a_local, blocksize);
        printf ("matrix b from %d\n", rank);
        //print_array (b_local, blocksize);
    //}

#endif /* */
    shmem_barrier_all ();
    //printf ("third barrier %d\n", rank);
    t2=0;
    tv[0] = gettime ();
    // start the matrix multiplication
    for( p = 1; p<=np; p++){
        tv2[0]=gettime();
    	for(i=0; i<ROWS; i++){
			for (k = 0; k < blocksize; k++) {
					for (j = 0; j < blocksize; j++) {
						c_local[i][j] = c_local[i][j] + a_local[i][k]
							* b_local[k + B_matrix_displacement][j];
					}
				}
	    }
        tv2[1]=gettime();
        t2 += dt(&tv2[1], &tv2[0]);
        shmem_barrier_all();
        if (rank == np - 1)
                shmem_double_put (&a_local[0][0], &a_local[0][0], blocksize*ROWS, 0);
	    else
                shmem_double_put (&a_local[0][0], &a_local[0][0], blocksize*ROWS,
                                  rank + 1);
        shmem_barrier_all();
    	if (B_matrix_displacement == 0)
    		B_matrix_displacement = (np - 1) * blocksize;
    	else
    		B_matrix_displacement = B_matrix_displacement - blocksize;
	
    }

    /*for (i = 0; i < ROWS; i++) {
        for (p = 1; p <= np; p++) {
        	tv2[0]=gettime();
            // compute the partial product of c[i][j]
            for (k = 0; k < blocksize; k++) {
                for (j = 0; j < blocksize; j++) {
                    c_local[i][j] = c_local[i][j] + a_local[i][k]
                        * b_local[k + B_matrix_displacement][j];
                }
            }
            tv2[1]=gettime();
            t2 += dt(&tv2[1], &tv2[0]);
            // send a block of matrix A to the adjacent PE
            shmem_barrier_all ();
            
            
            if (rank == np - 1)
                shmem_double_put (&a_local[i][0], &a_local[i][0], blocksize, 0);

            else
                shmem_double_put (&a_local[i][0], &a_local[i][0], blocksize,
                                  rank + 1);
            shmem_barrier_all ();
            
            // reset the displacement of matrix B to the next block
            if (B_matrix_displacement == 0)
                B_matrix_displacement = (np - 1) * blocksize;

            else
                B_matrix_displacement = B_matrix_displacement - blocksize;
        }
    }
    shmem_barrier_all ();
    //printf ("fourth barrier %d\n", rank);
    */
    tv[1] = gettime ();
    t = dt (&tv[1], &tv[0]);

#if DEBUG
    printf ("Process %d runtime: %4.2f Sec\n", rank, t / 1000000.0);

#endif /* */

    // Determine the maximum of the execution time for individual PEs
    shmem_double_max_to_all (&maxtime, &t, 1, 0, 0, size, pWrk, pSync);

    shmem_double_max_to_all (&maxtime2, &t2, 1, 0, 0, size, pWrk, pSync);

#if DEBUG                       // print the resultant array from root process
                                // if DEBUG enabled
    if (rank == 0) {
        printf ("matrix c from %d\n", rank);
        //print_array (c_local, blocksize);
    }

#endif /* */
    if (rank == 0) {
        printf ("Execution comp time in seconds =%.4f \n",  maxtime2 / 1000000.0);
        printf ("Execution comm time in seconds =%.4f \n",(maxtime-maxtime2) / 1000000.0);
        //printf("%f \n %f\n", c_local[1][10], c_local[10][100]);
        double timer[3];
        timer[0] = maxtime / 1000000.0;
        timer[2] = (maxtime-maxtime2)/ 1000000.0;
        timer[1] = maxtime2/ 1000000.0;
        print_time(3, timer);
    }
    return (0);
}
