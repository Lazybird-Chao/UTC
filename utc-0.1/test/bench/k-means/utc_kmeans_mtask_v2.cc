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
#include "../helper_getopt.h"

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
		/*if(getLrank()==0)
			std::cout<<"slave finish init()"<<std::endl;*/
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
		int changedObjs;
		int loopcounter = 0;
		Timer timer;
		double t1, t2, t3;
		t1 = t2 = t3=0;
		sbarrier.set(numTotalThreads);
		/* the main compute procedure */
		do{
			timer.start();
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
			//intra_Barrier();
			sbarrier.wait();
			t1 += timer.stop();
			/* send slave task's cluster info to master */
			/*cdt2master->WriteBy(0,&numChanges, sizeof(int), loopcounter*3);
			if(numTotalThreads>1)
				cdt2master->WriteBy(1,newClusterSize, numClusters*sizeof(int), loopcounter*3+1);
			else
				cdt2master->WriteBy(0,newClusterSize, numClusters*sizeof(int), loopcounter*3+1);
			if(numTotalThreads>2)
				cdt2master->WriteBy(2,newClusters[0], numClusters * numCoords*sizeof(float), loopcounter*3+2);
			else
				cdt2master->WriteBy(0,newClusters[0], numClusters * numCoords*sizeof(float), loopcounter*3+2);*/
			cdt2master->WriteByFirst(&numChanges, sizeof(int), loopcounter*3);
			cdt2master->WriteByFirst(newClusterSize, numClusters*sizeof(int), loopcounter*3+1);
			cdt2master->WriteByFirst(newClusters[0], numClusters * numCoords*sizeof(float), loopcounter*3+2);
			/* update local cluster from master */
			cdt2master->Read(&continueloop, sizeof(bool), loopcounter*2);

			if(continueloop){
				cdt2master->ReadByFirst(clusters[0], numClusters*numCoords*sizeof(float), loopcounter*2+1);
				/* reset newClusters for next loop */
				if(getUniqueExecution()){
					memset(newClusterSize, 0, numClusters*sizeof(int));
					memset(newClusters[0], 0, numClusters * numCoords*sizeof(float));
					numChanges = 0;
				}
				/*if(numTotalThreads>1 && taskThreadId ==1){
					memset(newClusterSize, 0, numClusters*sizeof(int));
					memset(newClusters[0], 0, numClusters * numCoords*sizeof(float));
					numChanges = 0;
				}
				else{
					memset(newClusterSize, 0, numClusters*sizeof(int));
					memset(newClusters[0], 0, numClusters * numCoords*sizeof(float));
					numChanges = 0;
				}*/
				//intra_Barrier();
				sbarrier.wait();
			}
			t2 += timer.stop();
			loopcounter++;
		}while(continueloop);

		/* finish compute */
		//double loopruntime = timer.stop();
		if(taskThreadId ==0){
			//std::cout<<"slave run() time: "<<loopruntime<<std::endl;
			//std::cout<<"loops: "<<loopcounter<<std::endl;
			runtime[0] = t1;
			runtime[1] = t2-t1;
			runtime[2] = t2;
		}

		/* send membership info to master */
		cdt2master->Write(membership, numObjs*sizeof(int), 0);

		free(localClusters[0]);
		free(localClusters);
		free(localClusterSize);

		//std::cout<<"slave finish run"<<std::endl;
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
	SpinBarrier sbarrier;
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
			goonloop = true;
			numObjsSlave = new int[cdt2slaves.size()];
		}
		intra_Barrier();
		/* get how many slave task instances */
		int numSlaves = cdt2slaves.size();

		/* averge dataset to slaves */
		datasetInfo datainfo2slave;
		int avgnumObjs = numObjs / numSlaves;
		int residue = numObjs % numSlaves;
		datainfo2slave.numClusters = numClusters;
		datainfo2slave.numCoords = numCoords;

		/* send initial dataset info to slaves */
		float* objptr = objects[0];
		for(int i=0; i< cdt2slaves.size(); i++){
			Conduit *cdt = cdt2slaves[i];
			if(i < residue){
				datainfo2slave.numObjs=avgnumObjs+1;
				numObjsSlave[i]=avgnumObjs+1;
			}
			else{
				datainfo2slave.numObjs = avgnumObjs;
				numObjsSlave[i]=avgnumObjs;
			}
			cdt->Write(&datainfo2slave, sizeof(datasetInfo), 0);
			int objsize = sizeof(float)*datainfo2slave.numObjs*numCoords;
			cdt->Write(objptr, objsize, 1);
			objptr= objptr + datainfo2slave.numObjs*numCoords;
			cdt->Write(clusters[0], sizeof(float)*numClusters*numCoords, 2);
		}
		/*if(getLrank()==0)
			std::cout<<"master finish init()"<<std::endl;*/

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
			Conduit *cdt;
			for(int i=cdt2slaves.size()-1; i>=0; i--){
				cdt = cdt2slaves[i];
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
				//intra_Barrier();
			}
			/* check if need do next loop */
			if(getLrank()==0){
				if(((float)numChanges)/numObjs > threshold && loopcounter < 100)
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
			//intra_Barrier();
			/* send new cluster info to slaves */
			for(int i=0; i< cdt2slaves.size(); i++){
				cdt = cdt2slaves[i];
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
		int *memberptr= &membership[0];
		for(int i=0; i< cdt2slaves.size(); i++){
			Conduit *cdt = cdt2slaves[i];
			cdt->Read(memberptr, sizeof(int)*numObjsSlave[i], 0);
			memberptr += numObjsSlave[i];
		}
		//std::cout<<"master finish run"<<std::endl;
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
	int *numObjsSlave;
	bool goonloop;

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
	char* filename = nullptr;
	float **objects;
	float **clusters;
	int *membership;
	int nthreads;
	int nslaves;
	int N;

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
					nslaves = atoi(optarg);
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
		if(filename == nullptr)
			usage(argv[0]);

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
	if(ctx.getProcRank()==0)
		std::cout<<"Start clustering..."<<std::endl;
	double master_runtime;
	double master_runtimeTotal=0;
	int loops=0;
	double slaves_runtime[4][3];
	double slaves_runtimeTotal[4][3];
	for(int i=0; i<4;i++)
		for(int j=0; j<3;j++)
			slaves_runtimeTotal[i][j]=0;
	Task<kmeans_master> kmeansComputeMaster("master", ProcList(0));
	std::vector<Task<kmeans_slave>*> kmeansComputeSlaves;
	std::vector<Conduit*> cdtMasterSlave;
	ProcList rlist(nthreads-1, 0);
	kmeansComputeSlaves.push_back(new Task<kmeans_slave>("slave",rlist));
	cdtMasterSlave.push_back(new Conduit(&kmeansComputeMaster, kmeansComputeSlaves[0]));
	for( int i=1; i< nslaves; i++){
		ProcList rlist(nthreads, i);
		kmeansComputeSlaves.push_back(new Task<kmeans_slave>("slave",rlist));
		cdtMasterSlave.push_back(new Conduit(&kmeansComputeMaster, kmeansComputeSlaves[i]));
	}

	for(int k=0; k<N; k++){
	if(ctx.getProcRank()==0){
			for (int i=0; i<numClusters; i++)
				for (int j=0; j<numCoords; j++)
					clusters[i][j] = objects[i][j];
		}
	ctx.Barrier();
	kmeansComputeMaster.init(objects,  numCoords,  numObjs,  numClusters,
			membership, clusters, cdtMasterSlave, &master_runtime, &loops);
	for(int i=0; i< nslaves; i++)
		kmeansComputeSlaves[i]->init(cdtMasterSlave[i], slaves_runtime[i]);

	kmeansComputeMaster.run();
	for(int i=0; i< nslaves; i++)
		kmeansComputeSlaves[i]->run();

	kmeansComputeMaster.wait();
	master_runtimeTotal+= master_runtime;
	for(int i=0; i< nslaves; i++){
		kmeansComputeSlaves[i]->wait();
		for(int j=0; j<3; j++)
			slaves_runtimeTotal[i][j]+=slaves_runtime[i][j];
	}
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
		std::cout<<"Master task run() time: "<<master_runtimeTotal/N<<std::endl;
		std::cout<<"Slave tasks run() time: "<<std::endl;
	}

	for(int i=0; i< nslaves; i++){
		if(ctx.getProcRank()==i){
			std::cout<<"\t"<<slaves_runtimeTotal[i][0]/N<<" "<<slaves_runtimeTotal[i][1]/N<<" "
				<<slaves_runtimeTotal[i][2]/N<<std::endl;
		}
		ctx.Barrier();
	}
	double avgtime[3];
	MPI_Reduce(slaves_runtimeTotal[ctx.getProcRank()], avgtime, 3, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	if(ctx.getProcRank()==0)
		std::cout<<"average: "<<avgtime[0]/(nslaves*N) <<" "<<avgtime[1]/(nslaves*N)<<
			" "<<avgtime[2]/(nslaves*N)<<std::endl;

	return 0;
}


