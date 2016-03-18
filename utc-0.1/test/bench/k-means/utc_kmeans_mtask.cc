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
struct datasetInfo{
	int numCoords;
	int numObjs;
	int numClusters;
};
class kmeans_slave{
public:
	void init(Conduit *cdt2master, double *runtime){
		struct datasetInfo datainfo;
		cdt2master->ReadBy(0, &datainfo, sizeof(datasetInfo), 0);
		cdt2master->ReadBy_Finish(0);
		if(getLrank()==0){
			numChanges =0;
			this->cdt2master = cdt2master;
			continueloop = true;
			this->runtime = runtime;

			numCoords = datainfo.numCoords;
			numObjs = datainfo.numObjs;
			numClusters = datainfo.numClusters;
			//std::cout<<"here"<<numClusters<<" "<<numCoords<<" "<<numObjs<<std::endl;
			objects    = (float**)malloc((numObjs) * sizeof(float*));
			assert(objects != NULL);
			objects[0] = (float*) malloc(numObjs * numCoords * sizeof(float));
			assert(objects[0] != NULL);
			for (int i=1; i<numObjs; i++)
				objects[i] = objects[i-1] + numCoords;
			clusters    = (float**) malloc(numClusters * sizeof(float*));
			assert(clusters != NULL);
			clusters[0] = (float*)  malloc(numClusters * numCoords * sizeof(float));
			assert(clusters[0] != NULL);
			for (int i=1; i<numClusters; i++)
				clusters[i] = clusters[i-1] + numCoords;
			membership = (int*) malloc(numObjs * sizeof(int));
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
		}
		intra_Barrier();
		/* get computation data from master task */
		cdt2master->Read(objects[0], numObjs * numCoords*sizeof(float), 1);
		cdt2master->Read(clusters[0], numClusters * numCoords * sizeof(float), 2);
		if(getLrank()==0)
			std::cout<<"slave finish init()"<<std::endl;
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
		for (int i=1; i<numClusters; i++)
			localClusters[i] = localClusters[i-1] + numCoords;
		int *localClusterSize = (int*) calloc(numClusters, sizeof(int));
		assert(localClusterSize != NULL);

		int localComputeSize = numObjs / numTotalThreads;
		int startObjIdx = localComputeSize*taskThreadId;
		int endObjIdx = startObjIdx + localComputeSize -1;
		int changedObjs;
		int loopcounter = 0;
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

			/* send slave task's cluster info to master */
			cdt2master->Write(&numChanges, sizeof(int), loopcounter*3);
			cdt2master->Write(newClusterSize, numClusters*sizeof(int), loopcounter*3+1);
			cdt2master->Write(newClusters[0], numClusters * numCoords*sizeof(float), loopcounter*3+2);
			/* update local cluster from master */
			cdt2master->Read(&continueloop, sizeof(bool), loopcounter*2);
			if(continueloop){
				cdt2master->Read(clusters[0], numClusters*numCoords*sizeof(float), loopcounter*2+1);
				/* reset newClusters for next loop */
				if(taskThreadId ==0){
					memset(newClusterSize, 0, numClusters*sizeof(int));
					memset(newClusters[0], 0, numClusters * numCoords*sizeof(float));
					numChanges = 0;
				}
				intra_Barrier();
			}
			loopcounter++;
		}while(continueloop);

		/* finish compute */
		double loopruntime = timer.stop();
		if(taskThreadId ==0){
			//std::cout<<"slave run() time: "<<loopruntime<<std::endl;
			//std::cout<<"loops: "<<loopcounter<<std::endl;
			*runtime = loopruntime;
		}

		/* send membership info to master */
		cdt2master->Write(membership, numObjs*sizeof(int), 0);

		free(localClusters[0]);
		free(localClusters);
		free(localClusterSize);
	}

	~kmeans_slave(){
			free(newClusters[0]);
			free(newClusters);
			free(newClusterSize);
			free(objects[0]);
			free(objects);
			free(clusters[0]);
			free(clusters);
			free(membership);
		}

private:
	float euclid_dist_2(int    numdims,  /* no. dimensions */
		                    float *coord1,   /* [numdims] */
		                    float *coord2)   /* [numdims] */
	{
		int i;
		float ans=0.0;

		for (i=0; i<numdims; i++){
			ans += (coord1[i]-coord2[i]) * (coord1[i]-coord2[i]);
		}
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

	Conduit *cdt2master;
	bool continueloop;

	double *runtime;
};

class kmeans_master{
public:
	void init(float** objects, int numCoords, int numObjs, int numClusters,
			int *membership, float **clusters, std::vector<Conduit*> cdt2slaves,
			double *runtime, int *loops){
		if(getLrank()==0){
			this->runtime = runtime;
			this->loops = loops;
			this->objects = objects;
			this->numCoords = numCoords;
			this->numObjs = numObjs;
			this->numClusters = numClusters;
			this->membership = membership;
			this->clusters = clusters;
			threshold = 0.0001;
			numChanges =0;
			this->cdt2slaves = cdt2slaves;
			newClusters    = (float**) malloc(numClusters * sizeof(float*));
			assert(newClusters != NULL);
			newClusters[0] = (float*)  calloc(numClusters * numCoords, sizeof(float));
			assert(newClusters[0] != NULL);
			for (int i=1; i<numClusters; i++)
				newClusters[i] = newClusters[i-1] + numCoords;
			newClusterSize = (int*) calloc(numClusters, sizeof(int));
			assert(newClusterSize != NULL);
			clusterSize = (int*) calloc(numClusters, sizeof(int));


			/* get how many slave task instances */
			int numSlaves = cdt2slaves.size();
			/* averge dataset to slaves */
			datainfo2slave.numObjs = numObjs / numSlaves;
			datainfo2slave.numClusters = numClusters;
			datainfo2slave.numCoords = numCoords;
			goonloop = true;
		}
		intra_Barrier();
		/* send initial dataset info to slaves */
		for(int i=0; i< cdt2slaves.size(); i++){
			Conduit *cdt = cdt2slaves[i];
			cdt->Write(&datainfo2slave, sizeof(datasetInfo), 0);
			float* objptr = objects[0 + i*datainfo2slave.numObjs];
			int objsize = sizeof(float)*datainfo2slave.numObjs*numCoords;
			cdt->Write(objptr, objsize, 1);
			cdt->Write(clusters[0], sizeof(float)*numClusters*numCoords, 2);
		}
		if(getLrank()==0)
			std::cout<<"master finish init()"<<std::endl;

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
		int startObjIdx = localComputeSize*taskThreadId;
		int endObjIdx = startObjIdx + localComputeSize -1;
		int loopcounter =0;
		Timer timer;
		timer.start();
		/* the main compute procedure */
		do{
			if(getLrank()==0){
				memset(clusters[0], 0, sizeof(float)*numClusters*numCoords);
				memset(clusterSize, 0, sizeof(int)*numClusters);
			}
			/* read new cluster info from slaves */
			for(int i=0; i< cdt2slaves.size(); i++){
				Conduit *cdt = cdt2slaves[i];
				cdt->Read(&changedObjs, sizeof(int), loopcounter*3);
				cdt->Read(newClusterSize, sizeof(int)*numClusters, loopcounter*3+1);
				cdt->Read(newClusters[0], sizeof(float)*numClusters*numCoords, loopcounter*3+2);

				if(getLrank()==0){
					numChanges +=changedObjs;
					for(int j=0; j<numClusters; j++){
						for(int k=0; k<numCoords; k++){
							clusters[j][k]+=newClusters[j][k];
						}
						clusterSize[j]+=newClusterSize[j];
					}
				}
				intra_Barrier();
			}
			/* check if need do next loop */
			if(getLrank()==0){
				if(((float)numChanges)/numObjs > threshold && loopcounter < 500)
					goonloop = true;
				else
					goonloop = false;
				numChanges=0;
				/* update cluster */
				for(int i=0; i<numClusters; i++){
					for(int j=0; j<numCoords; j++){
						if(clusterSize[i] >0)
							clusters[i][j]=clusters[i][j]/clusterSize[i];
					}
				}
			}
			intra_Barrier();
			/* send new cluster info to slaves */
			for(int i=0; i< cdt2slaves.size(); i++){
				Conduit *cdt = cdt2slaves[i];
				cdt->Write(&goonloop, sizeof(bool), loopcounter*2);
				if(goonloop)
					cdt->Write(clusters[0], sizeof(float)*numClusters*numCoords, loopcounter*2+1);
			}

			loopcounter++;
		}while(goonloop);

		/* finish compute */
		double loopruntime = timer.stop();
		if(taskThreadId ==0){
			//std::cout<<"master run() time: "<<loopruntime<<std::endl;
			//std::cout<<"loops: "<<loopcounter<<std::endl;
			*runtime = loopruntime;
			*loops = loopcounter;
		}

		/* get membership info */
		for(int i=0; i< cdt2slaves.size(); i++){
			Conduit *cdt = cdt2slaves[i];
			int *memberptr = &membership[i*datainfo2slave.numObjs];
			cdt->Read(memberptr, sizeof(int)*datainfo2slave.numObjs, 0);
		}

	}

	~kmeans_master(){
		free(newClusters[0]);
		free(newClusters);
		free(newClusterSize);
		free(clusterSize);
	}

private:

	float **objects;      /* in: [numObjs][numCoords] */
	int     numCoords;    /* no. features */
	int     numObjs;      /* no. objects */
	int     numClusters;  /* no. clusters */
	float   threshold;    /* percent of objects change membership */
	int    *membership;   /* out: [numObjs] */
	float **clusters;     /* out: [numClusters][numCoords] */
	int numChanges;
	int changedObjs;
	float **newClusters;
	int *newClusterSize;
	int *clusterSize;

	std::vector<Conduit*> cdt2slaves;
	datasetInfo datainfo2slave;
	bool goonloop;

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
	int nslaves;
	if(argc<5){
		std::cout<<"run like: ./a.out 'nthread' 'nslaves'  'num-cluster' 'inputfile'"<<std::endl;
	}
	else{
		nthreads = atoi(argv[1]);
		nslaves = atoi(argv[2]);
		numClusters = atoi(argv[3]);
		filename = argv[4];
	}
	/* startup utc contex*/
	UtcContext ctx(argc, argv);

	/* read data points from file*/
	//std::cout<<"reading data points from file."<<std::endl;
	Task<FileRead> file_read("file-read", ProcList(0));
	file_read.init(filename, &numObjs, &numCoords, std::ref(objects));
	file_read.run();
	file_read.finish();

	/* allocate a 2D space for clusters[] (coordinates of cluster centers)
	       this array should be the same across all processes                  */
	if(ctx.getProcRank()==0){
		clusters = (float**) malloc(numClusters * sizeof(float*));
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
	//std::cout<<"Start clustering..."<<std::endl;
	double master_runtime;
	int loops=0;
	double slaves_runtime[32];
	Task<kmeans_master> kmeansComputeMaster("master", ProcList(0));
	std::vector<Task<kmeans_slave>*> kmeansComputeSlaves;
	std::vector<Conduit*> cdtMasterSlave;
    for( int i=0; i< nslaves; i++){
    	ProcList rlist(nthreads, i);
    	kmeansComputeSlaves.push_back(new Task<kmeans_slave>("slave",rlist));
    	cdtMasterSlave.push_back(new Conduit(&kmeansComputeMaster, kmeansComputeSlaves[i]));
    }

	kmeansComputeMaster.init(objects,  numCoords,  numObjs,  numClusters,
			membership, clusters, cdtMasterSlave, &master_runtime, &loops);
	for(int i=0; i< nslaves; i++)
		kmeansComputeSlaves[i]->init(cdtMasterSlave[i], &slaves_runtime[i]);

	kmeansComputeMaster.run();
	for(int i=0; i< nslaves; i++)
		kmeansComputeSlaves[i]->run();

	kmeansComputeMaster.wait();
	for(int i=0; i< nslaves; i++)
			kmeansComputeSlaves[i]->wait();



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
	for(auto& slave: kmeansComputeSlaves)
		delete slave;
	for(auto& cdt: cdtMasterSlave)
		delete cdt;

	/* output some info*/
	if(ctx.getProcRank() == 0){
		printf("Input file:     %s\n", filename);
		printf("numObjs       = %d\n", numObjs);
		printf("numCoords     = %d\n", numCoords);
		printf("numClusters   = %d\n", numClusters);
		std::cout<<"loops: "<<loops<<std::endl;
		std::cout<<"Master task run() time: "<<master_runtime<<std::endl;
		std::cout<<"Slave tasks run() time: "<<std::endl;
	}
	for(int i=0; i< nslaves; i++){
		if(ctx.getProcRank()==i)
			std::cout<<"\t"<<slaves_runtime[i]<<std::endl;
		ctx.Barrier();
	}
	return 0;
}


