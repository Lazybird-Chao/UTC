/*
 * utc_kmeans.cc
 *
 *  Created on: Mar 2, 2016
 *      Author: chaoliu
 */

/* user included file*/
#include "file_io.h"

/* main UTC header file */
#include "Utc.h"

/* other standard header file */
#include <iostream>
#include <string>
#include <cstdio>
#include <cstring>
#include <cstdlib>

using namespace iUtc;

class kmeans_algorithm{
public:
	void init(float** objects, int numCoords, int numObjs, int numClusters,
			float *membership, float **clusters){
		this->objects = objects;
		this->numCoords = numCoords;
		this->numObjs = numObjs;
		this->numClusters = numClusters;
		this->membership = numClusters;
		this->clusters = clusters;
		threshold = 1;
	}

	void run(){

	}

private:
	float **objects;      /* in: [numObjs][numCoords] */
	int     numCoords;    /* no. features */
	int     numObjs;      /* no. objects */
	int     numClusters;  /* no. clusters */
	float   threshold;    /* % objects change membership */
	int    *membership;   /* out: [numObjs] */
	float **clusters;     /* out: [numClusters][numCoords] */
};


int main(int argc, char*argv[]){
	int numClusters, numCoords, numObjs;
	char* filename, *center_filename;
	float **objects;
	float **clusters;
	int *membership;

	if(argc<3){
		std::cout<<"run like: ./a.out 'num-cluster' 'inputfile'"<<std::endl;
	}
	else{
		numClusters = atoi(argv[1]);
		filename = argv[2];
	}
	/* startup utc contex*/
	UtcContext ctx(argc, argv);

	/* read data points from file*/
	std::cout<<"reading data points from file."<<std::endl;
	objects = file_read(0, filename, &numObjs, &numCoords);

	/* allocate a 2D space for clusters[] (coordinates of cluster centers)
	       this array should be the same across all processes                  */
	clusters    = (float**) malloc(numClusters *             sizeof(float*));
	assert(clusters != NULL);
	clusters[0] = (float*)  malloc(numClusters * numCoords * sizeof(float));
	assert(clusters[0] != NULL);
	for (int i=1; i<numClusters; i++)
		clusters[i] = clusters[i-1] + numCoords;

	/* initial cluster centers with first numClusters points*/
	for (int i=0; i<numClusters; i++)
		for (int j=0; j<numCoords; j++)
			clusters[i][j] = objects[i][j];

	/* begin clustering */
	std::cout<<"Start clustering..."<<std::endl;




	/* write cluster centers to output file*/
	file_write(filename, numClusters, numObjs, numCoords, clusters,
	               membership, 0);
	free(membersip);
	free(clusters[0]);
	free(clusters);
	free(objects[0]);
	free(objects);

	/* output some info*/
	printf("Input file:     %s\n", filename);
	printf("numObjs       = %d\n", numObjs);
	printf("numCoords     = %d\n", numCoords);
	printf("numClusters   = %d\n", numClusters);

	return 0;
}


