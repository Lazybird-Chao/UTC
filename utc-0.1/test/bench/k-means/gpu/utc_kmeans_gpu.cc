/*
 * utc_kmeans_gpu.cc
 *
 *  Created on: Nov 7, 2016
 *      Author: chao
 *
 *
 *      utc-gpu version of kmeans
 *      work for single GPU
 *
 *      This program only use GPU to compute the new membership
 *      of each objects.
 *      After get the membership, we compute new clusters and check
 *      the number of changed objects in host-part with CPU
 *
 */

#include "../file_io.h"
#include "../../helper_getopt.h"
#include "../../helper_printtime.h"
#include "utc_kmeans_gpu_kernel.h"

#include "Utc.h"
#include "UtcGpu.h"

#include "iostream"

using namespace iUtc;

class kmeansGPU:public UserTaskBase{
private:
	float **objects;
	int	numCoords;
	int numObjs;
	int numClusters;
	float threshold;
	int *membership;
	float **clusters;
	int numChanges;

	float **newClusters;
	int *newClusterSize;
	int *newMembership;

	float *objects_d;
	int *membership_d;
	//float *newMembership_d;
	float *clusters_d;

	//float *newClusters_d;
	//float *newClusterSize_d;

	int blocksize;
	int batchPerThread;

public:
	void initImpl(float** objects, int numCoords, int numObjs,
			int numClusters,
			int *membership,
			float **clusters,
			int blocksize,
			int batchPerThread){
		std::cout<<"begin init ..."<<std::endl;
		if(__localThreadId ==0){
			threshold = 0.0001;
			numChanges = 0;

			this->objects = objects;
			this->numCoords = numCoords;
			this->numObjs = numObjs;
			this->numClusters = numClusters;
			this->membership = membership;
			this->clusters = clusters;

			newClusters    = (float**) malloc(this->numClusters * sizeof(float*));
			assert(newClusters != NULL);

			newClusters[0] = (float*)  calloc(numClusters*numCoords, sizeof(float));
			assert(newClusters[0] != NULL);

			for (int i=1; i<this->numClusters; i++)
				newClusters[i] = newClusters[i-1] + this->numCoords;

			newClusterSize = (int*) calloc(this->numClusters, sizeof(int));

			for(int i=0; i<this->numObjs; i++)
				this->membership[i]=-1;

			newMembership = (int*)malloc(this->numObjs*sizeof(int));

			this->blocksize = blocksize;
			this->batchPerThread = batchPerThread;

		}

		inter_Barrier();
		if(__localThreadId ==0){
			std::cout<<"task: "<<getCurrentTask()->getName()<<" finish initImpl on "
					<<__processId<<std::endl;
		}
	}

	void runImpl(double *runtime){
		Timer t1, t2;
		std::cout<<"begin run ..."<<std::endl;

		t1.start();
		checkCudaRuntimeErrors(
				cudaMalloc(&objects_d, numObjs*numCoords*sizeof(float)));
		checkCudaRuntimeErrors(
				cudaMalloc(&clusters_d, numClusters*numCoords*sizeof(float)));
		checkCudaRuntimeErrors(
				cudaMalloc(&membership_d, numObjs*sizeof(int)));
		runtime[0] = t1.stop();

		t1.start();
		t2.start();
		checkCudaRuntimeErrors(
				cudaMemcpy(objects_d, objects[0], numObjs*numCoords*sizeof(float), cudaMemcpyHostToDevice));
		//checkCudaRuntimeErrors(
		//		cudaMemcpy(clusters_d, clusters[0], numClusters*numCoords*sizeof(float), cudaMemcpyHostToDevice));
		//checkCudaRuntimeErrors(
		//		cudaMemcpy(membership_d, membership, numObjs*sizeof(int), cudaMemcpyHostToDevice));
		runtime[1] = t1.stop();

		GpuKernel mykernel;
		mykernel.setBlockDim(blocksize);
		mykernel.setGridDim((numObjs+blocksize*batchPerThread -1)/(blocksize*batchPerThread));
		mykernel.setNumArgs(7);
		mykernel.setArgs<float*>(0, objects_d);
		mykernel.setArgs<float*>(1, clusters_d);
		/*float clusters_dd[150];
		for(int i=0; i< 30; i++){
			for(int j=0; j< 5; j++)
				clusters_dd[i*5+j]=clusters[i][j];
		}
		mykernel.setArgs<float[150]>(1, clusters_dd);*/
		mykernel.setArgs<int*>(2, membership_d);
		mykernel.setArgs<int>(3, numObjs);
		mykernel.setArgs<int>(4, numClusters);
		mykernel.setArgs<int>(5, numCoords);
		mykernel.setArgs<int>(6, batchPerThread);

		int changedObjs = 0;
		int loopcounter =0;
		do{
			t1.start();
			checkCudaRuntimeErrors(
					cudaMemcpy(clusters_d, clusters[0], numClusters*numCoords*sizeof(float), cudaMemcpyHostToDevice));
			runtime[1] += t1.stop();

			t1.start();
			mykernel.launchKernel((const void*)&computeNewMembership_kernel);
			runtime[2] += t1.stop();

			t1.start();
			checkCudaRuntimeErrors(
					cudaMemcpy(newMembership, membership_d, numObjs*sizeof(int),cudaMemcpyDeviceToHost));
			runtime[3] += t1.stop();

			// record changes and comput new clusters
			t1.start();
			changedObjs = 0;
			for(int i=0; i<numObjs; i++){
				if(newMembership[i] != membership[i]){
					changedObjs++;
					membership[i] = newMembership[i];
				}
				//std::cout<<newMembership[i]<<std::endl;
				newClusterSize[newMembership[i]]++;
				for(int j=0; j<numCoords; j++)
					newClusters[newMembership[i]][j] += objects[i][j];
			}
			//std::cout<<ERROR_LINE<<std::endl;
			for(int i=0; i<numClusters; i++){
				for(int j=0; j<numCoords; j++){
					if(newClusterSize[i]>0)
						clusters[i][j] = newClusters[i][j]/newClusterSize[i];
					newClusters[i][j] = 0;
				}
				newClusterSize[i] = 0;
			}
			runtime[4]+=t1.stop();
			/*for(int i=0; i< 30; i++){
				for(int j=0; j< 5; j++)
					clusters_dd[i*5+j]=clusters[i][j];
			}*/
		}while(((float)changedObjs)/numObjs > threshold && loopcounter++ < 100);
		runtime[5] = t2.stop();

		inter_Barrier();
		if(__localThreadId ==0){
			std::cout<<"End loopcounter: "<<loopcounter<<std::endl;
			std::cout<<"task: "<<getCurrentTask()->getName()<<" finish run on "
					<<__processId<<std::endl;
		}
	}

	~kmeansGPU(){
		if(newClusters){
			free(newClusters[0]);
			free(newClusters);
			free(newClusterSize);
			free(newMembership);
		}

		if(objects_d){
			cudaFree(objects_d);
			cudaFree(membership_d);
			cudaFree(clusters_d);
		}
	}
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
int main(int argc, char* argv[]){
	int numClusters, numCoords, numObjs;
	char* filename;
	float **objects;
	float **clusters;
	int *membership;
	int nthreads;
	int nprocs;
	int N;

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
	int procs = ctx.numProcs();
	if(nprocs != procs){
		std::cerr<<"process number not match with arguments '-p' !!!\n";
		return 1;
	}
	int myproc = ctx.getProcRank();
	if(nthreads != 1){
		std::cerr<<"only run with 1 thread for this program\n";
		return 1;
	}


	/* read data points from file*/
	if(ctx.getProcRank()==0)
		std::cout<<"reading data points from file."<<std::endl;
	Task<FileRead> file_read("file-read", ProcList(0));
	file_read.init(filename, &numObjs, &numCoords, std::ref(objects));
	file_read.run();
	file_read.finish();

	if(ctx.getProcRank()==0){
		std::cout<<numObjs<<" "<<numCoords<<" "<<numClusters<<std::endl;
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

	double runtime[6]={0,0,0,0,0,0};
	Task<kmeansGPU> myKmeans(ProcList(0), TaskType::gpu_task);
	myKmeans.init(objects, numCoords, numObjs, numClusters,
			membership, clusters, 256, 1);
	myKmeans.run(runtime);
	myKmeans.wait();

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

	if(myproc==0){
		std::cout<<"total time: "<<runtime[5]<<std::endl;
		std::cout<<"mem alloc time: "<<runtime[0]<<std::endl;
		std::cout<<"mem copyin time: "<<runtime[1]<<std::endl;
		std::cout<<"kernel run time: "<<runtime[2]<<std::endl;
		std::cout<<"mem copyout time: "<<runtime[3]<<std::endl;
		std::cout<<"comp new cluster time: "<<runtime[4]<<std::endl;

		print_time(6, runtime);
	}
}



