/*
 * mpi_heat_image.cc
 *
 *  Created on: Oct 18, 2016
 *      Author: chao
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <sys/time.h>
#include <unistd.h>
#include "../helper_printtime.h"


// dimension.h
#define DIM2( basetype, name, w1 ) basetype (*name)[w1]
#define DIM3( basetype, name, w1, w2 ) basetype (*name)[w1][w2]
#define DIM4( basetype, name, w1, w2, w3 ) basetype (*name)[w1][w2][w3]
#define DIM5( basetype, name, w1, w2, w3, w4 ) basetype (*name)[w1][w2][w3][w4]
#define DIM6( basetype, name, w1, w2, w3, w4, w5 ) basetype (*name)[w1][w2][w3][w4][w5]
#define DIM7( basetype, name, w1, w2, w3, w4, w5, w6 ) basetype (*name)[w1][w2][w3][w4][w5][w6]
// file name of output image
#define FILENAME ".image"
// Change here the number of steps, the cell geometry, etc
#define NITER 5000
#define STEPITER 1000
#define delx 0.5
#define dely 0.25
// end change here.

void itstep (int mx, int my, void *pf, void *pnewf, void *pr,
		double rdx2,double rdy2, double beta)
{
    DIM2 (double, f, my) = (typeof (f)) pf;
    DIM2 (double, newf, my) = (typeof (newf)) pnewf;
    DIM2 (double, r, my) = (typeof (r)) pr;
    int i, j, mx1, my1;
    mx1 = mx - 1;
    my1 = my - 1;
    for (i = 1; i < mx1; i++) {
        for (j = 1; j < my1; j++) {
            newf[i][j] =
                ((f[i - 1][j] + f[i + 1][j]) * rdx2 +
                 (f[i][j - 1] + f[i][j + 1]) * rdy2 - r[i][j]) * beta;
        }
    }
}

int main(int argc, char**argv){
	int i, j, n, mx1, mx2, my1, my_number, n_of_nodes,
		totalmx, partmx, leftmx, mx, my;
	FILE *fp;
	double t, tv[2];
	double rdx2, rdy2, beta;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_number);
	MPI_Comm_size(MPI_COMM_WORLD, &n_of_nodes);

	if (argc != 3) {
		if (!my_number)
			fprintf (stderr, "Usage: %s <nrows> <ncolumns>\n", argv[0]);
		return (-1);
	}

	totalmx = mx = (int)atoi(argv[1]);
	my = (int) atoi(argv[2]);
	if (my < 1) {
		if (!my_number)
			fprintf (stderr, "Number of columns (%d) should be positive\n", my);
		return (-1);
	}

/* Compute the number of rows per node: */
	mx = (totalmx + n_of_nodes - 1) / n_of_nodes;

/* This is the number of rows for all but the last: */
	partmx = mx;

/* This is the number of rows for the last: */
/* It cannot be greater than partmx, but it can be non-positive: */
	leftmx = totalmx - partmx * (n_of_nodes - 1);
	if (leftmx < 1) {
		if (!my_number)
			fprintf (stderr, "Cannot distribute rows, too many processors\n");
		return (-1);
	}

	if (my_number == (n_of_nodes - 1))
		mx = leftmx;
/* End rows distribution. */
	partmx += 2;
	mx += 2;
	my += 2;

/* Here we know the array sizes, so make the arrays themselves: */
	{
		DIM2 (double, f, my);
		DIM2 (double, newf, my);
		DIM2 (double, r, my);
		typeof (f) pf[2];
		int curf;

		f = (typeof (f)) malloc (2 * partmx * sizeof (*f));
		r = (typeof (r)) malloc (mx * sizeof (*r));
		if ((!f) || (!r)) {
			fprintf (stderr, "Cannot allocate, exiting\n");
			exit (-1);
		}
		curf = 0;
		pf[0]=f;
		pf[1]=f+partmx;
		newf = pf[1];
		rdx2 = 1. / delx / delx;
		rdy2 = 1. / dely / dely;
		beta = 1.0 / (2.0 * (rdx2 + rdy2));
		if (!my_number) {
			printf
				("Solving heat conduction task on %d by %d grid by %d processors\n",
				 totalmx, my - 2, n_of_nodes);
			fflush (stdout);
		}

		for (i = 0; i < mx; i++) {
			for (j = 0; j < my; j++) {
				if (((i == 0) && (my_number == 0)) || (j == 0)
					|| ((i == (mx - 1)) && (my_number == (n_of_nodes - 1)))
					|| (j == (my - 1)))
					newf[i][j] = f[i][j] = 1.0;
				else
					newf[i][j] = f[i][j] = 0.0;
				r[i][j] = 0.0;
			}
		}

		mx1 = mx - 1;
		my1 = my - 1;
		MPI_Barrier(MPI_COMM_WORLD);

/* Iteration loop: */
		int imageCount = 1;
		int k;
		char filename[20];
		char fidx[2];
		fidx[1]='\0';
		double t, tcomp, tcomm;
		t= tcomp = tcomm =0;
		double tv1[2];
		for(k=0; k<imageCount; k++){
			fidx[0]='0'+k;
			strcat(filename,fidx);
			strcpy(filename, FILENAME);
			tv[0] = MPI_Wtime();
			for(n=0; n<NITER; n++){
				if (!my_number) {
					if (!(n % STEPITER))
						printf ("Iteration %d\n", n);
				}
				/* Step of calculation starts here: */
				f = pf[curf];
				newf = pf[1 - curf];
				tv1[0]=MPI_Wtime();
				itstep (mx, my, f, newf, r, rdx2, rdy2, beta);
				tv1[1]=MPI_Wtime();
				tcomp += tv1[1]-tv1[0];
				/* Do all the transfers: */
				MPI_Barrier(MPI_COMM_WORLD);
				tv1[0] = MPI_Wtime();

				/* send the lower bound */
				if(my_number<(n_of_nodes-1))
					MPI_Send(&newf[mx-2][1], my-2, MPI_DOUBLE, my_number+1, my_number, MPI_COMM_WORLD);
				if(my_number>0)
					MPI_Recv(&newf[0][1], my-2, MPI_DOUBLE, my_number-1, my_number-1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				/* send the upper bound */
				if(my_number>0)
					MPI_Send(&newf[1][1], my-2, MPI_DOUBLE, my_number-1, my_number, MPI_COMM_WORLD);
				if(my_number<(n_of_nodes-1))
					MPI_Recv(&newf[partmx-1][1], my-2, MPI_DOUBLE, my_number+1, my_number+1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

				MPI_Barrier(MPI_COMM_WORLD);
				tv1[1]=MPI_Wtime();
				tcomm+=tv1[1] - tv1[0];

				/* swap the halves: */
				curf = 1 - curf;

			}

			if (!my_number) {
				tv[1] = MPI_Wtime();
				t = tv[1] - tv[0];
				printf ("Elapsed time: %4.2f sec\n", t);
				printf ("comp time: %4.2f sec\n", tcomp);
				printf ("comm time: %4.2f sec\n", tcomm);
				printf ("Output image file in current directory\n");
				fp = fopen (filename, "w");
				fclose (fp);

				double timer[3];
				timer[0] = t;
				timer[1] = tcomp;
				timer[2] = tcomm;
				print_time(3,timer);
			}

			/*
			for (j = 0; j < n_of_nodes; j++) {
				MPI_Barrier(MPI_COMM_WORLD);
				if (j == my_number) {
					fp = fopen (filename, "a");
					for (i = 1; i < (mx - 1); i++)
						fwrite (&(newf[i][1]), my - 2, sizeof (newf[0][0]), fp);
					fclose (fp);
				}
			}
			*/
		}

		//free(f);
		//free(r);
	}

	MPI_Finalize();

	return 0;

}
