//-------------------------------------------------------------------------//
//                                                                         //
//  This benchmark is a serial C version of the NPB FT code. This C        //
//  version is developed by the Center for Manycore Programming at Seoul   //
//  National University and derived from the serial Fortran versions in    //
//  "NPB3.3-SER" developed by NAS.                                         //
//                                                                         //
//  Permission to use, copy, distribute and modify this software for any   //
//  purpose with or without fee is hereby granted. This software is        //
//  provided "as is" without express or implied warranty.                  //
//                                                                         //
//  Information on NPB 3.3, including the technical report, the original   //
//  specifications, source code, results and information on how to submit  //
//  new results, is available at:                                          //
//                                                                         //
//           http://www.nas.nasa.gov/Software/NPB/                         //
//                                                                         //
//  Send comments or suggestions for this C version to cmp@aces.snu.ac.kr  //
//                                                                         //
//          Center for Manycore Programming                                //
//          School of Computer Science and Engineering                     //
//          Seoul National University                                      //
//          Seoul 151-744, Korea                                           //
//                                                                         //
//          E-mail:  cmp@aces.snu.ac.kr                                    //
//                                                                         //
//-------------------------------------------------------------------------//

//-------------------------------------------------------------------------//
// Authors: Sangmin Seo, Jungwon Kim, Jun Lee, Jeongho Nah, Gangwon Jo,    //
//          and Jaejin Lee                                                 //
//-------------------------------------------------------------------------//

//---------------------------------------------------------------------
// FT benchmark
//---------------------------------------------------------------------

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include "global.h"
#include "print_results.h"


static char getclass();

logical timers_enabled;
/////
dcomplex *plane;
dcomplex **scr;
double ***twiddle;
dcomplex ***xnt;
dcomplex ***y;

int main(int argc, char *argv[])
{
	/////////
	int i, j;
	plane = (dcomplex*)malloc((BLOCKMAX+1)*MAXDIM * sizeof(dcomplex));

	scr = (dcomplex**)malloc(MAXDIM *sizeof(dcomplex*));
	scr[0] = (dcomplex*)malloc((BLOCKMAX+1)*MAXDIM *sizeof(dcomplex));
	for( i=1; i<MAXDIM; i++)
		scr[i]=scr[i-1]+ (BLOCKMAX+1);

	twiddle= (double***)malloc(NZ * sizeof(double**));
	twiddle[0]=(double**)malloc(NZ * NY * sizeof(double*));
	for( i=1; i<NZ; i++)
		twiddle[i]=twiddle[i-1]+ NY;
	twiddle[0][0]=(double*)malloc(NZ*NY*(NX+1)*sizeof(double));
	for(i=1; i<NY; i++)
		twiddle[0][i] = twiddle[0][i-1] + (NX+1);
	for(i=1; i<NZ; i++){
		twiddle[i][0] = twiddle[i-1][NY-1]+ (NX+1);
		for( j=1; j<NY; j++)
			twiddle[i][j] = twiddle[i][j-1]+ (NX+1);
	}

	xnt= (dcomplex***)malloc(NZ * sizeof(dcomplex**));
	xnt[0]=(dcomplex**)malloc(NZ * NY * sizeof(dcomplex*));
	for( i=1; i<NZ; i++)
		xnt[i]=xnt[i-1]+ NY;
	xnt[0][0]=(dcomplex*)malloc(NZ*NY*(NX+1)*sizeof(dcomplex));
	for(i=1; i<NY; i++)
		xnt[0][i] = xnt[0][i-1] + (NX+1);
	for( i=1; i<NZ; i++){
		xnt[i][0] = xnt[i-1][NY-1]+ (NX+1);
		for( j=1; j<NY; j++)
			xnt[i][j] = xnt[i][j-1]+ (NX+1);
	}

	y= (dcomplex***)malloc(NZ * sizeof(dcomplex**));
	y[0]=(dcomplex**)malloc(NZ * NY * sizeof(dcomplex*));
	for( i=1; i<NZ; i++)
		y[i]=y[i-1]+ NY;
	y[0][0]=(dcomplex*)malloc(NZ*NY*(NX+1)*sizeof(dcomplex));
	for( i=1; i<NY; i++)
		y[0][i] = y[0][i-1] + (NX+1);
	for( i=1; i<NZ; i++){
		y[i][0] = y[i-1][NY-1]+ (NX+1);
		for( j=1; j<NY; j++)
			y[i][j] = y[i][j-1]+ (NX+1);
	}
	////////////
  int niter;
  char Class;
  double total_time, mflops;
  logical verified;

  FILE *fp;
  if ((fp = fopen("timer.flag", "r")) != NULL) {
    timers_enabled = true;
    fclose(fp);
  } else {
    timers_enabled = false;
  }

  niter = NITER_DEFAULT;

  printf("\n\n NAS Parallel Benchmarks (NPB3.3-SER-C) - FT Benchmark\n\n");
  printf(" Size                : %4dx%4dx%4d\n", NX, NY, NZ);
  printf(" Iterations          :     %10d\n", niter);
  printf("\n");

  Class = getclass();

  appft(niter, &total_time, &verified);

  if (total_time != 0.0) {
    mflops = 1.0e-6 * (double)NTOTAL *
            (14.8157 + 7.19641 * log((double)NTOTAL)
             + (5.23518 + 7.21113 * log((double)NTOTAL)) * niter)
            / total_time;
  } else {
    mflops = 0.0;
  }

  print_results("FT", Class, NX, NY, NZ, niter,
                total_time, mflops, "          floating point", verified, 
                NPBVERSION, COMPILETIME, CS1, CS2, CS3, CS4, 
                CS5, CS6, CS7);

  return 0;
}


static char getclass()
{
  if ((NX == 64) && (NY == 64) &&                 
      (NZ == 64) && (NITER_DEFAULT == 6)) {
    return 'S';
  } else if ((NX == 128) && (NY == 128) &&
             (NZ == 32) && (NITER_DEFAULT == 6)) {
    return 'W';
  } else if ((NX == 256) && (NY == 256) &&
             (NZ == 128) && (NITER_DEFAULT == 6)) {
    return 'A';
  } else if ((NX == 512) && (NY == 256) &&
             (NZ == 256) && (NITER_DEFAULT == 20)) {
    return 'B';
  } else if ((NX == 512) && (NY == 512) &&
             (NZ == 512) && (NITER_DEFAULT == 20)) {
    return 'C';
  } else if ((NX == 2048) && (NY == 1024) &&
             (NZ == 1024) && (NITER_DEFAULT == 25)) {
    return 'D';
  } else {
    return 'U';
  }
}

