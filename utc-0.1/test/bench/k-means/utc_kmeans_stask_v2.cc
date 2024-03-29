/*
 * utc_kmeans.cc
 *
 *  Created on: Mar 2, 2016
 *      Author: chaoliu
 */

/* user included file*/
#include "file_io.h"
#include "../helper_getopt.h"

/* main UTC header file */
#include "Utc.h"

/* other standard header file */
#include <iostream>
#include <string>
#include <cstdio>
#include <cstring>
#include <cstdlib>

using namespace iUtc;

class kmeans_algorithm : public UserTaskBase{
public:
	void initImpl(float** objects, int numCoords, int numObjs, int numClusters,
			int *membership, float **clusters, double *runtime, int *loops){
		if(getLrank()==0){
			this->runtime = runtime;
			this->loops = loops;
			threshold = 0.0001;
			numChanges =0;
		}
		if(getGrank()==0){
			this->objects = objects;
			this->numCoords = numCoords;
			this->numObjs = numObjs;
			this->numClusters = numClusters;
			this->membership = membership;
			this->clusters = clusters;
			//std::cout<<getGrank()<<" "<<this->numCoords<<" "<<this->numObjs<<" "<<this->numClusters<<std::endl;
		}
		SharedDataBcast((void*)&(this->numCoords), sizeof(int), 0);
		SharedDataBcast((void*)&(this->numObjs), sizeof(int), 0);
		SharedDataBcast((void*)&(this->numClusters), sizeof(int), 0);
		//std::cout<<getGrank()<<" "<<this->numCoords<<" "<<this->numObjs<<" "<<this->numClusters<<std::endl;
		int objsize = this->numCoords * this->numObjs;
		int clustersize = this->numClusters *this->numCoords;
		if(getGrank()!=0 && getLrank()==0){
			this->objects = (float**)malloc(this->numObjs * sizeof(float*));
			this->objects[0]=(float*)malloc(objsize*sizeof(float));
			for(int i=1; i<this->numObjs; i++)
				this->objects[i]=this->objects[i-1]+this->numCoords;
			this->clusters    = (float**) malloc(this->numClusters * sizeof(float*));
			assert(this->clusters != NULL);
			this->clusters[0] = (float*)  malloc(clustersize*sizeof(float));
			assert(this->clusters[0] != NULL);
			for (int i=1; i<numClusters; i++)
				this->clusters[i] = this->clusters[i-1] + this->numCoords;
			this->membership = (int*)malloc(this->numObjs *sizeof(int));
		}
		intra_Barrier();
		SharedDataBcast((void*)this->objects[0], objsize*sizeof(float), 0);
		SharedDataBcast((void*)this->clusters[0], clustersize*sizeof(float), 0);
		//std::cout<<getGrank()<<std::endl;
		if(getLrank()==0){
			newClusters    = (float**) malloc(this->numClusters * sizeof(float*));
			assert(newClusters != NULL);
			newClusters[0] = (float*)  calloc(clustersize, sizeof(float));
			assert(newClusters[0] != NULL);
			for (int i=1; i<this->numClusters; i++)
				newClusters[i] = newClusters[i-1] + this->numCoords;
			newClusterSize = (int*) calloc(this->numClusters, sizeof(int));
			for(int i=0; i<this->numObjs; i++){
				this->membership[i]=-1;
			}
		}
		if(getLrank()==0)
			std::cout<<"Finish init."<<std::endl;
	}

	void runImpl(){
		int taskThreadId = getGrank();
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
	    float *clusterforgather;
	    int *changeforgather;
	    int *clustersizeforgather;
	    if(taskThreadId ==0){
	    	clusterforgather = (float*)malloc(numClusters* numCoords * getPsize()*sizeof(float));
	    	changeforgather = new int[getPsize()];
	    	clustersizeforgather = (int*)malloc(numClusters*getPsize()*sizeof(int));
	    }

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
		double loopruntime[2];
		loopruntime[0]=loopruntime[1]=0;

		/* the main compute procedure */
		do{
			changedObjs =0;
			timer.start();
			/* find each object's belonged cluster */
			for(int i= startObjIdx; i<= endObjIdx; i++){
				int idx = find_nearest_cluster(numClusters, numCoords, objects[i], clusters);
				/*int idx=0;
				float min_dist = 0.0;
				for(int j=0; j<numCoords; j++){
					min_dist +=(objects[i][j] - clusters[0][j])*(objects[i][j] - clusters[0][j]);
				}
				float dist = 0.0;
				for(int k=1; k<numClusters; k++){
					dist =0.0;
					for(int j=0; j<numCoords; j++){
						dist +=(objects[i][j] - clusters[k][j])*(objects[i][j] - clusters[k][j]);
					}
					if(dist < min_dist){
						min_dist = dist;
						idx = k;
					}
				}*/

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
			updateNewCluster.lock();
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
			updateNewCluster.unlock();
			/* sync all task-threads, wait local compute finish*/
			intra_Barrier();
			loopruntime[0]+=timer.stop();
			//std::cout<<"changes1:"<<numChanges<<std::endl;
			/* get total changes*/
			SharedDataGather(&numChanges, sizeof(int), changeforgather, 0);
			if(taskThreadId==0){
				numChanges=0;
				for(int i=0; i<getPsize();i++){
					numChanges+=changeforgather[i];
				}
			}
			SharedDataBcast(&numChanges,sizeof(int), 0);
			changedObjs = numChanges;
			//std::cout<<"changes2:"<<changedObjs<<std::endl;
			/* get global clusters */
			SharedDataGather(newClusters[0], numClusters*numCoords*sizeof(float),
										clusterforgather, 0);
			SharedDataGather(newClusterSize, numClusters*sizeof(int),
										clustersizeforgather, 0);
			if(taskThreadId ==0){
				for(int i=1; i<getPsize();i++){
					for(int j=0; j<numClusters; j++){
						newClusterSize[j] += clustersizeforgather[i*numClusters + j];
						for(int k=0; k<numCoords; k++){
							newClusters[j][k] += clusterforgather[i*numClusters*numCoords+
																  j*numCoords+k];
						}
					}
				}
			}
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
			else if(getLrank()==0){
				for(int i=0; i<numClusters; i++){
					newClusterSize[i]=0;
					for(int j=0; j<numCoords; j++)
						newClusters[i][j]=0;
				}
				numChanges =0;
			}
			SharedDataBcast(clusters[0], numClusters*numCoords*sizeof(float), 0);
			loopruntime[1]+= timer.stop();
			//loopcounter++;
		}while(((float)changedObjs)/numObjs > threshold && loopcounter++ < 100);

		/* finish compute */
		if(getLrank() ==0){
			//std::cout<<"run() time: "<<loopruntime<<std::endl;
			//std::cout<<"changed objs:"<<changedObjs<<std::endl;
			//std::cout<<"loops: "<<loopcounter<<std::endl;
			runtime[0] = loopruntime[0];
			runtime[1] = loopruntime[1]-loopruntime[0];
			runtime[2] = loopruntime[1];
			*loops = loopcounter;
		}
		free(localClusters[0]);
		free(localClusters);
		free(localClusterSize);
		if(taskThreadId==0){
			free(clusterforgather);
			delete changeforgather;
			free(clustersizeforgather);
		}
		if(getGrank()!=0 && getLrank()==0){
			free(objects[0]);
			free(objects);
			free(clusters[0]);
			free(clusters);
			free(membership);
		}
		/*if(getLrank()==0)
			std::cout<<"Finish run."<<std::endl;*/
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

	//SharedDataLock updateNewCluster;
	SpinLock updateNewCluster;


	double *runtime;
	int *loops;
};

void usage(char *argv0) {
    std::string help =
        "Usage: %s [switches] -i filename\n"
        "       -i filename     :  file containing data to be clustered\n"
        "       -t nthreads     :threads per node of Task\n"
		"       -p nprocs       :number of nodes running on \n"
        "       -n clusters     :number of clusters\n"
    	"       -l nloops       :number of loops to run the test\n";
    fprintf(stderr, help.c_str(), argv0);
    exit(-1);
}

int main(int argc, char*argv[]){
	int numClusters, numCoords, numObjs;
	char* filename;
	float **objects;
	float **clusters;
	int *membership;
	int nthreads;
	int nprocs;
	int N;

	/* startup utc contex*/
	UtcContext &ctx = UtcContext::getContext(argc, argv);

	int opt;
	extern char *optarg;
	extern int optind;
	opt=getopt(argc, argv, "i:t:p:n:l:");
	while( opt!=EOF ){
		switch (opt){
			case 'i':
				filename = optarg;
				break;
			case 't':
				nthreads = atoi(optarg);
				break;
			case 'p':
				nprocs = atoi(optarg);
				break;
			case 'n':
				numClusters = atoi(optarg);
				break;
			case 'l':
				N = atoi(optarg);
				break;
			case '?':
				usage(argv[0]);
				break;
			default:
				usage(argv[0]);
				break;
		}
		//std::cout<<opt;
		opt=getopt(argc, argv, "i:t:p:n:l:");
	}

	/* read data points from file*/
	if(ctx.getProcRank()==0)
		std::cout<<"reading data points from file."<<std::endl;
	Task<FileRead> file_read("file-read", ProcList(0));
	file_read.init(filename, &numObjs, &numCoords, std::ref(objects));
	file_read.run();
	file_read.finish();

	/* allocate a 2D space for clusters[] (coordinates of cluster centers)
	       this array should be the same across all processes                  */
	if(ctx.getProcRank()==0){
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
	}
	/* begin clustering */
	if(ctx.getProcRank()==0)
		std::cout<<"Start clustering..."<<std::endl;
	double kmeans_runtime[3];
	double kmeans_runtimeTotal[3];
	kmeans_runtimeTotal[0]=kmeans_runtimeTotal[1]=kmeans_runtimeTotal[2]=0;
	int loops=0;
	std::vector<int> rvec;
	for(int i=0; i< nprocs; i++)
		for(int j=0; j<nthreads;j++)
			rvec.push_back(i);
	ProcList rlist(rvec); // task with  nthreads on each proc
	Task<kmeans_algorithm> kmeansCompute("kmeans", rlist);
	for(int k =0; k<N; k++){
		if(ctx.getProcRank()==0){
		for (int i=0; i<numClusters; i++)
					for (int j=0; j<numCoords; j++)
						clusters[i][j] = objects[i][j];
		}
	ctx.Barrier();
	kmeansCompute.init(objects,  numCoords,  numObjs,  numClusters,
			membership, clusters, kmeans_runtime, &loops);
	kmeansCompute.run();
	kmeansCompute.wait();
	kmeans_runtimeTotal[0]+=kmeans_runtime[0];
	kmeans_runtimeTotal[1]+=kmeans_runtime[1];
	kmeans_runtimeTotal[2]+=kmeans_runtime[2];
	ctx.Barrier();
	}


	/* write cluster centers to output file*/
	Task<FileWrite> file_write("file-write", ProcList(0));
	file_write.init(filename, numClusters, numObjs, numCoords, clusters,
            membership, 0);
	file_write.run();
	file_write.finish();

	if(ctx.getProcRank()==0){
		free(membership);
		free(clusters[0]);
		free(clusters);
		free(objects[0]);
		free(objects);
	}

	/* output some info*/
	double t1, t2, t3;
	MPI_Reduce(&kmeans_runtimeTotal[0], &t1, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&kmeans_runtimeTotal[1], &t2, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&kmeans_runtimeTotal[2], &t3, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	if(ctx.getProcRank()==0){
		printf("Input file:     %s\n", filename);
		printf("numObjs       = %d\n", numObjs);
		printf("numCoords     = %d\n", numCoords);
		printf("numClusters   = %d\n", numClusters);
		std::cout<<"loops: "<<loops<<std::endl;
		std::cout<<"task run() time: "<<t1/N<<
				" "<<t2/N<<
				" "<<t3/N<<std::endl;
	}
	return 0;
}


