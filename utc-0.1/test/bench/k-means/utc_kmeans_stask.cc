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
			int *membership, float **clusters, double *runtime, int *loops){
		if(getLrank()==0){
			this->runtime = runtime;
			this->objects = objects;
			this->numCoords = numCoords;
			this->numObjs = numObjs;
			this->numClusters = numClusters;
			this->membership = membership;
			this->clusters = clusters;
			this->loops = loops;
			threshold = 0.0001;
			numChanges =0;
			for (int i=0; i<numObjs; i++)
				membership[i] = -1;
			newClusters    = (float**) malloc(numClusters * sizeof(float*));
			assert(newClusters != NULL);
			newClusters[0] = (float*)  calloc(numClusters * numCoords, sizeof(float));
			assert(newClusters[0] != NULL);
			for (int i=1; i<numClusters; i++)
				newClusters[i] = newClusters[i-1] + numCoords;
			newClusterSize = (int*) calloc(numClusters, sizeof(int));
			assert(newClusterSize != NULL);
			std::cout<<"finish init()"<<std::endl;
		}

	}

	void run(){
		int taskThreadId = getTrank();
		int numTotalThreads = getGsize();
		/* local cluster centers used in each thread*/
		float **localClusters;
		localClusters    = (float**) malloc(numClusters * sizeof(float*));
		assert(localClusters != NULL);
		localClusters[0] = (float*)  calloc(numClusters * numCoords, sizeof(float));
		assert(localClusters[0] != NULL);
		/* recored each cluster's size */
		for (int i=1; i<numClusters; i++)
			localClusters[i] = localClusters[i-1] + numCoords;
		int *localClusterSize = (int*) calloc(numClusters, sizeof(int));
	    assert(localClusterSize != NULL);

		int localComputeSize = numObjs / numTotalThreads;
		int residue = numObjs%numTotalThreads;
		int startObjIdx;
		int endObjIdx;
		if(taskThreadId < residue){
			startObjIdx = (localComputeSize+1)*taskThreadId;
			endObjIdx = startObjIdx + localComputeSize;
		}
		else{
			startObjIdx = (localComputeSize+1)*residue + localComputeSize*(taskThreadId-residue);
			endObjIdx = startObjIdx + localComputeSize -1;
		}
		int loopcounter =0;
		int changedObjs;
		Timer timer;
		timer.start();
		/* the main compute procedure */
		do{
			changedObjs =0;
			/* find each object's belonged cluster */
			for(int i= startObjIdx; i<= endObjIdx; i++){
				int idx = find_nearest_cluster(numClusters, numCoords, objects[i], clusters);
				/* check if membership changed */
				if(idx != membership[i]){
					changedObjs++;
				}
				membership[i] = idx;
				/* record this obj in cluster[idx] */
				localClusterSize[idx]++;
				for(int j=0; j<numCoords; j++)
					localClusters[idx][j] += objects[i][j];
			}
			/* update task-threads shared newClusters with localClusters */
			updateNewCluster.write_lock();
			for(int i=0; i<numClusters; i++){
				if(localClusterSize[i] != 0){
					for(int j=0; j<numCoords; j++){
						newClusters[i][j] += localClusters[i][j];
						localClusters[i][j] = 0;
					}
					newClusterSize[i]+=localClusterSize[i];
					localClusterSize[i]=0;
				}
			}
			numChanges +=changedObjs;
			updateNewCluster.write_unlock();

			/* sync all task-threads, wait local compute finish*/
			intra_Barrier();
			/* get total changes*/
			changedObjs = numChanges;
			intra_Barrier();
			/* update cluster with newClusters for next loop */
			if(taskThreadId ==0){
				for(int i=0; i<numClusters; i++){
					for(int j=0; j<numCoords; j++){
						if(newClusterSize[i] >0)
							clusters[i][j]=newClusters[i][j]/newClusterSize[i];
						newClusters[i][j]=0;
					}
					newClusterSize[i]=0;
				}
				numChanges =0;
			}
			intra_Barrier();
			loopcounter++;
		}while(((float)changedObjs)/numObjs > threshold && loopcounter < 500);

		/* finish compute */
		double loopruntime = timer.stop();
		if(taskThreadId ==0){
			//std::cout<<"run() time: "<<loopruntime<<std::endl;
			//std::cout<<"changed objs:"<<changedObjs<<std::endl;
			//std::cout<<"loops: "<<loopcounter<<std::endl;
			*runtime = loopruntime;
			*loops = loopcounter;
		}
		free(localClusters[0]);
		free(localClusters);
		free(localClusterSize);
	}

	~kmeans_algorithm(){
		free(newClusters[0]);
		free(newClusters);
		free(newClusterSize);
	}

private:
	float euclid_dist_2(int    numdims,  /* no. dimensions */
	                    float *coord1,   /* [numdims] */
	                    float *coord2)   /* [numdims] */
	{
	    int i;
	    float ans=0.0;

	    for (i=0; i<numdims; i++)
	        ans += (coord1[i]-coord2[i]) * (coord1[i]-coord2[i]);
	    //ans = sqrt(ans);
	    return(ans);
	}

	int find_nearest_cluster(int     numClusters, /* no. clusters */
	                         int     numCoords,   /* no. coordinates */
	                         float  *object,      /* [numCoords] */
	                         float **clusters)    /* [numClusters][numCoords] */
	{
	    int   index, i;
	    float dist, min_dist;

	    /* find the cluster id that has min distance to object */
	    index    = 0;
	    min_dist = euclid_dist_2(numCoords, object, clusters[0]);

	    for (i=1; i<numClusters; i++) {
	        dist = euclid_dist_2(numCoords, object, clusters[i]);
	        /* no need square root */
	        if (dist < min_dist) { /* find the min and its array index */
	            min_dist = dist;
	            index    = i;
	        }
	    }
	    return(index);
	}

	float **objects;      /* in: [numObjs][numCoords] */
	int     numCoords;    /* no. features */
	int     numObjs;      /* no. objects */
	int     numClusters;  /* no. clusters */
	float   threshold;    /* percent of objects change membership */
	int    *membership;   /* out: [numObjs] */
	float **clusters;     /* out: [numClusters][numCoords] */
	int numChanges;
	float **newClusters;
	int *newClusterSize;
	SharedDataLock updateNewCluster;

	double *runtime;
	int *loops;
};


int main(int argc, char*argv[]){
	int numClusters, numCoords, numObjs;
	char* filename;
	float **objects;
	float **clusters;
	int *membership;
	int nthreads;
	if(argc<4){
		std::cout<<"run like: ./a.out 'nthread' 'num-cluster' 'inputfile'"<<std::endl;
	}
	else{
		nthreads = atoi(argv[1]);
		numClusters = atoi(argv[2]);
		filename = argv[3];
	}
	/* startup utc contex*/
	UtcContext ctx(argc, argv);

	/* read data points from file*/
	std::cout<<"reading data points from file."<<std::endl;
	Task<FileRead> file_read("file-read", ProcList(0));
	file_read.init(filename, &numObjs, &numCoords, std::ref(objects));
	file_read.run();
	file_read.finish();

	/* allocate a 2D space for clusters[] (coordinates of cluster centers)
	       this array should be the same across all processes                  */
	clusters    = (float**) malloc(numClusters *             sizeof(float*));
	assert(clusters != NULL);
	clusters[0] = (float*)  malloc(numClusters * numCoords * sizeof(float));
	assert(clusters[0] != NULL);
	for (int i=1; i<numClusters; i++)
		clusters[i] = clusters[i-1] + numCoords;
	/* allocate space for membership array for each object */
	membership = (int*) malloc(numObjs * sizeof(int));
	assert(membership != NULL);
	/* initial cluster centers with first numClusters points*/
	for (int i=0; i<numClusters; i++)
		for (int j=0; j<numCoords; j++)
			clusters[i][j] = objects[i][j];

	/* begin clustering */
	std::cout<<"Start clustering..."<<std::endl;
	double kmeans_runtime;
	int loops=0;
	ProcList rlist(nthreads, 0); // task with  nthreads on proc 0
	Task<kmeans_algorithm> kmeansCompute("kmeans", rlist);
	kmeansCompute.init(objects,  numCoords,  numObjs,  numClusters,
			membership, clusters, &kmeans_runtime, &loops);
	kmeansCompute.run();
	kmeansCompute.wait();
	//kmeansCompute.finish();


	/* write cluster centers to output file*/
	Task<FileWrite> file_write("file-write", ProcList(0));
	file_write.init(filename, numClusters, numObjs, numCoords, clusters,
            membership, 0);
	file_write.run();
	file_write.finish();

	free(membership);
	free(clusters[0]);
	free(clusters);
	free(objects[0]);
	free(objects);

	/* output some info*/
	printf("Input file:     %s\n", filename);
	printf("numObjs       = %d\n", numObjs);
	printf("numCoords     = %d\n", numCoords);
	printf("numClusters   = %d\n", numClusters);
	std::cout<<"loops: "<<loops<<std::endl;
	std::cout<<"task run() time: "<<kmeans_runtime<<std::endl;

	return 0;
}


