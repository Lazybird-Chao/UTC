/*
 * mpi_matrix.cc
 *
 *  Created on: Oct 17, 2016
 *      Author: chao
 *
 *      Divide matrix along columns, and use mpi send/recv to
 *      exchange data with neighbours
 */
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <sys/time.h>
#include <unistd.h>
#include "../helper_printtime.h"

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

double maxtime;
double t, tv[2];
double t2, tv2[2];
double maxtime2;

int main(int argc, char** argv){
	MPI_Init(&argc, &argv);
	int i, j, k;
	int blocksize;
	int rank, size, nextpe;
	int p, np;                  // round and number of process
	double **a_local, **b_local;
	double **c_local;
	int B_matrix_displacement;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	np = size;

	if(argc == 2 ){
		COLUMNS = ROWS = atoi(argv[1]);
	}

	blocksize = COLUMNS / np;
	B_matrix_displacement = rank*blocksize;

	MPI_Barrier(MPI_COMM_WORLD);

	a_local = (double **) malloc (ROWS * sizeof (double *));
	a_local[0]= (double *) malloc (ROWS * blocksize * sizeof (double));
	b_local = (double **) malloc (ROWS * sizeof (double *));
	b_local[0] = (double *) malloc (ROWS * blocksize * sizeof (double));
	c_local = (double **) malloc (ROWS * sizeof (double *));
	c_local[0] = (double *) malloc (ROWS * blocksize * sizeof (double));

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
	if (rank == 0) {
		printf ("matrix a from %d %d\n", rank, blocksize);
		//print_array (a_local, blocksize);
		printf ("matrix b from %d\n", rank);
		//print_array (b_local, blocksize);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	double *a_tmp;
	if(np>1){
		a_tmp = (double*)malloc(ROWS*blocksize*sizeof(double));
	}
	t2=0;
	tv[0]=MPI_Wtime();
	// compute local matrix
	tv2[0] = MPI_Wtime();
	for(i=0; i<ROWS; i++){
		for(k=0; k<blocksize; k++){
			for(j=0; j<blocksize; j++){
				c_local[i][j] = c_local[i][j] + a_local[i][k]
							* b_local[k + B_matrix_displacement][j];
			}
		}
	}
	// exchange data with other proc and comput
	tv2[1]= MPI_Wtime();
	t2+=tv2[1] - tv2[0];
	int neighbour = rank;
	for(p=2; p<=np; p++){
		neighbour = (neighbour+1)%np;
		if(rank ==np-1){
			MPI_Recv(a_tmp, blocksize*ROWS, MPI_DOUBLE, neighbour, neighbour, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Send(&a_local[0][0], blocksize*ROWS, MPI_DOUBLE, neighbour, rank, MPI_COMM_WORLD);
			//memcpy(a_local[0], a_tmp,blocksize*ROWS*sizeof(double));
		}
		else{
			MPI_Send(&a_local[0][0], blocksize*ROWS, MPI_DOUBLE, neighbour, rank, MPI_COMM_WORLD);
			MPI_Recv(a_tmp, blocksize*ROWS, MPI_DOUBLE, neighbour, neighbour, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		B_matrix_displacement = neighbour * blocksize;


		MPI_Barrier(MPI_COMM_WORLD);

		tv2[0] = MPI_Wtime();
		for(i=0; i<ROWS; i++){
			for(k=0; k<blocksize; k++){
				for(j=0; j<blocksize; j++){
					c_local[i][j] = c_local[i][j] + a_tmp[i*blocksize+k]
								* b_local[k + B_matrix_displacement][j];
				}
			}
		}
		tv2[1]= MPI_Wtime();
		t2+=tv2[1] - tv2[0];
		//MPI_Barrier(MPI_COMM_WORLD);

	}
	tv[1] = MPI_Wtime();
	t = tv[1]-tv[0];

	double runtime=0;
	MPI_Reduce(&t, &runtime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	runtime /= np;
	double comptime = 0;
	MPI_Reduce(&t2, &comptime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	comptime /= np;

	if (rank == 0) {
		printf ("Execution comp time in seconds =%.4f \n",  comptime);
		printf ("Execution comm time in seconds =%.4f \n", runtime - comptime);
		//printf("%f \n %f\n", c_local[1][10], c_local[10][100]);
		double timer[3];
		timer[0] = runtime;
		timer[2] = runtime - comptime;
		timer[1] = comptime;;
		print_time(3, timer);
	}
	free(a_local[0]);
	free(a_local);
	free(b_local[0]);
	free(b_local);
	free(c_local[0]);
	free(c_local);

	MPI_Finalize();
}

