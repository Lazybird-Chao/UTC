/*
 * file_io.h
 *
 *  Created on: Mar 2, 2016
 *      Author: chaoliu
 */

#ifndef TEST_BENCH_K_MEANS_FILE_IO_H_
#define TEST_BENCH_K_MEANS_FILE_IO_H_

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*   File:         file_io.c                                                 */
/*   Description:  This program reads point data from a file                 */
/*                 and write cluster output to files                         */
/*   Input file format:                                                      */
/*                 ascii  file: each line contains 1 data object             */
/*                 binary file: first 4-byte integer is the number of data   */
/*                 objects and 2nd integer is the no. of features (or        */
/*                 coordinates) of each object                               */
/*                                                                           */
/*   Author:  Wei-keng Liao                                                  */
/*            ECE Department Northwestern University                         */
/*            email: wkliao@ece.northwestern.edu                             */
/*                                                                           */
/*   Copyright (C) 2005, Northwestern University                             */
/*   See COPYRIGHT notice in top-level directory.                            */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>     /* strtok() */
#include <sys/types.h>  /* open() */
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>     /* read(), close() */
#include <errno.h>
#include <cassert>
extern int errno;
int _debug =0;

#define MAX_CHAR_PER_LINE 128
#include "Utc.h"
using namespace iUtc;

/*---< file_read() >---------------------------------------------------------*/
class FileRead{
public:
	void init(    char *filename,      /* input file name */
                  int  *numObjs,       /* no. data objects (local) */
                  int  *numCoords,     /* no. coordinates */
				  float **&objects_out)
	{
		if(getLrank() == 0){
			this->isBinaryFile = isBinaryFile;
			this->filename = filename;
			this->numObjs = numObjs;
			this->numCoords = numCoords;

			int len;
			if ((infile = fopen(filename, "r")) == NULL) {
				fprintf(stderr, "Error: no such file (%s)\n", filename);
				return;
			}
			/* first find the number of objects */
			lineLen = MAX_CHAR_PER_LINE;
			line = (char*) malloc(lineLen);
			assert(line != NULL);

			(*numObjs) = 0;
			while (fgets(line, lineLen, infile) != NULL) {
				/* check each line to find the max line length */
				while (strlen(line) == lineLen-1) {
					/* this line read is not complete */
					len = strlen(line);
					fseek(infile, -len, SEEK_CUR);

					/* increase lineLen */
					lineLen += MAX_CHAR_PER_LINE;
					line = (char*) realloc(line, lineLen);
					assert(line != NULL);

					ret = fgets(line, lineLen, infile);
					assert(ret != NULL);
				}

				if (strtok(line, " \t\n") != 0)
					(*numObjs)++;
			}
			rewind(infile);
			if (_debug) printf("lineLen = %d\n",lineLen);

			/* find the no. coordinates of each object */
			(*numCoords) = 0;
			while (fgets(line, lineLen, infile) != NULL) {
				if (strtok(line, " \t\n") != 0) {
					/* ignore the id (first coordiinate): numCoords = 1; */
					while (strtok(NULL, " ,\t\n") != NULL) (*numCoords)++;
					break; /* this makes read from 1st object */
				}
			}
			rewind(infile);
			if (_debug) {
				printf("File %s numObjs   = %d\n",filename,*numObjs);
				printf("File %s numCoords = %d\n",filename,*numCoords);
			}

			/* allocate space for objects[][] and read all objects */
			len = (*numObjs) * (*numCoords);
			objects_out    = (float**)malloc((*numObjs) * sizeof(float*));
			assert(objects != NULL);
			objects_out[0] = (float*) malloc(len * sizeof(float));
			assert(objects_out[0] != NULL);
			for (int i=1; i<(*numObjs); i++)
				objects_out[i] = objects_out[i-1] + (*numCoords);
			this->objects = objects_out;
		}
	}

	void run(){
		int     i, j;

		if(getLrank() == 0){

			i = 0;
			/* read all objects */
			while (fgets(line, lineLen, infile) != NULL) {
				if (strtok(line, " \t\n") == NULL) continue;
				for (j=0; j<(*numCoords); j++) {
					objects[i][j] = atof(strtok(NULL, " ,\t\n"));
					if (_debug && i == 0) /* print the first object */
						printf("object[i=%d][j=%d]=%f\n",i,j,objects[i][j]);
				}
				i++;
			}
			assert(i == *numObjs);

			fclose(infile);
			free(line);
		}
	}

private:
	int isBinaryFile;
	char *filename;
	int *numObjs;
	int *numCoords;
	float **objects;

	FILE *infile;
	char *line, *ret;
	int   lineLen;
};


/*---< file_write() >---------------------------------------------------------*/
class FileWrite{
public:
	void init(char      *filename,     /* input file name */
            int        numClusters,  /* no. clusters */
            int        numObjs,      /* no. data objects */
            int        numCoords,    /* no. coordinates (local) */
            float    **clusters,     /* [numClusters][numCoords] centers */
            int       *membership,   /* [numObjs] */
            int        verbose)
	{
		if(getLrank() == 0){
			this->filename = filename;
			this->numClusters = numClusters;
			this->numObjs = numObjs;
			this->numCoords = numCoords;
			this->clusters = clusters;
			this->membership = membership;
			this->verbose = verbose;
		}
	}

	void run(){
		if(getLrank() == 0){
			FILE *fptr;
		    int   i, j;
		    char  outFileName[1024];

		    /* output: the coordinates of the cluster centres ----------------------*/
		    sprintf(outFileName, "%s.cluster_centres", filename);
		    if (verbose) printf("Writing coordinates of K=%d cluster centers to file \"%s\"\n",
		                        numClusters, outFileName);
		    fptr = fopen(outFileName, "w");
		    for (i=0; i<numClusters; i++) {
		        fprintf(fptr, "%d ", i);
		        for (j=0; j<numCoords; j++)
		            fprintf(fptr, "%f ", clusters[i][j]);
		        fprintf(fptr, "\n");
		    }
		    fclose(fptr);

		    /* output: the closest cluster centre to each of the data points --------*/
		    sprintf(outFileName, "%s.membership", filename);
		    if (verbose) printf("Writing membership of N=%d data objects to file \"%s\"\n",
		                        numObjs, outFileName);
		    fptr = fopen(outFileName, "w");
		    for (i=0; i<numObjs; i++)
		        fprintf(fptr, "%d %d\n", i, membership[i]);
		    fclose(fptr);
		}

	}
private:
	char *filename;
	int numClusters;
	int numObjs;
	int numCoords;
	float **clusters;
	int *membership;
	int verbose;
};



#endif /* TEST_BENCH_K_MEANS_FILE_IO_H_ */
