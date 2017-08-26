/*
 * task.cc
 *
 *  Created on: Mar 22, 2017
 *      Author: chao
 */

#include "task.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>     /* strtok() */
#include <sys/types.h>  /* open() */
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>     /* read(), close() */

template <typename T>
void FileRead<T>::runImpl(int isBinaryFile,
			char* filename,
			T **objects,
			int *numObjs,
			int *numCoords){
	float *objs;
	int     i, j, len;
	size_t numBytesRead;
	if(isBinaryFile){
		int infile;
		//fprintf(stderr, "Trying to read from binary file: %s", filename);
		if ((infile = open(filename, O_RDONLY, "0600")) == -1) {
			fprintf(stderr, "Error: Input File Not Found\n");
			exit(EXIT_FAILURE);
		}
		numBytesRead = read(infile, numObjs, sizeof(int));
		assert(numBytesRead == sizeof(int));
		numBytesRead = read(infile, numCoords, sizeof(int));
		assert(numBytesRead == sizeof(int));

		len = (*numObjs) * (*numCoords);
		objs = (float*)malloc(len * sizeof(float));
		numBytesRead = read(infile, objs, len*sizeof(float));
		assert(numBytesRead == len*sizeof(float));
		close(infile);
	}
	else{
		FILE *infile;
		char *line, *ret;
		int   lineLen;
		if ((infile = fopen(filename, "r")) == NULL) {
			fprintf(stderr, "Error: Input File Not Found\n");
			exit(EXIT_FAILURE);
		}
		/* first find the number of objects */
		int MAX_CHAR_PER_LINE = 256;
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
		/* find the no. objects of each object */
		(*numCoords) = 0;
		while (fgets(line, lineLen, infile) != NULL) {
			if (strtok(line, " \t\n") != 0) {
				/* ignore the id (first coordiinate): numCoords = 1; */
				while (strtok(NULL, " ,\t\n") != NULL) (*numCoords)++;
				break; /* this makes read from 1st object */
			}
		}
		rewind(infile);

		/* allocate space for objects[][] and read all objects */
		len = (*numObjs) * (*numCoords);
		objs = (float*)malloc(len * sizeof(float));
		/* read all objects */
		i=0;
		while (fgets(line, lineLen, infile) != NULL) {
			if (strtok(line, " \t\n") == NULL) continue;
			for (j=0; j<(*numCoords); j++)
				objs[i*(*numCoords) +j] = (float)atof(strtok(NULL, " ,\t\n"));
			i++;
		}
		fclose(infile);
		free(line);
	}
	if(sizeof(T) == sizeof(float))
		*objects = (T*)objs;
	else{
		*objects = (T*)malloc((*numObjs) * (*numCoords)* sizeof(double));
		for (i=0; i< (*numObjs); i++){
			for (j=0; j<(*numCoords); j++){
				(*objects)[i*(*numCoords) +j] = objs[i*(*numCoords) +j];
			}
		}
		free(objs);
	}
	return;
}

template class FileRead<float>;
template class FileRead<double>;


template<typename T>
void Output<T>::runImpl(char* filename, T *objsNN, int numNN, int numCoords){
	int l;
	FILE* fp = fopen(filename, "w");
	for(int j = 0; j < numNN; j++) {
		fprintf(fp, "Neighbor %d: ", j);
		for(l = 0; l < numCoords; l++)
			fprintf(fp, "%f ", objsNN[j*numCoords + l]);
		fprintf(fp, "\n");
	}
	fclose(fp);
}
template class Output<float>;
template class Output<double>;




