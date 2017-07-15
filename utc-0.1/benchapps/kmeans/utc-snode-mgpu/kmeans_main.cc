/*
 * kmeans_main.cc
 *
 *  Created on: Mar 22, 2017
 *      Author: chao
 */

#include <iostream>
#include <iomanip>

#include "../../common/helper_getopt.h"
#include "../../common/helper_err.h"
#include "../../common/helper_printtime.h"
#include "Utc.h"
#include "UtcGpu.h"
using namespace iUtc;

#include "task.h"
#include "mgpu/kmeans_task_mgpu.h"

#define FTYPE float

int main(int argc, char**argv){
	bool printTime = false;
	int     isBinaryFile = 0;
	char   *filename = NULL;
	char *outfile = NULL;
	FTYPE threshold = 0.01;
	int numClusters = 1;

	int nthreads=1;
	int nprocess=1;

	MemType memtype = MemType::pageable;
	int mtype = 0;
	/* initialize UTC context */
	UtcContext &ctx = UtcContext::getContext(argc, argv);

	/* Parse command line options */
	int     opt;
	extern char   *optarg;
	extern int     optind;
	 while ( (opt=getopt(argc,argv,"a:o:i:n:bvt:p:m:"))!= EOF) {
		switch (opt) {
			case 'v': printTime = true;
					  break;
			case 't': nthreads=atoi(optarg);
				  break;
			case 'p': nprocess = atoi(optarg);
				  break;
			case 'm': mtype = atoi(optarg);
				  break;
			case 'i': filename=optarg;
					  break;
			case 'b': isBinaryFile = 1;
					  break;
			case 'n': numClusters = atoi(optarg);
					  break;
			case 'o': outfile = optarg;
					  break;
			case 'a': threshold = (FTYPE)atof(optarg);
					  break;
			case '?':
				break;
			default:
				break;
		}
	}
	if(filename == NULL){
		std::cerr<<"Need input file path !!!"<<std::endl;
		return 1;
	}

	int procs = ctx.numProcs();
	int myproc = ctx.getProcRank();
	if(nprocess != procs){
		std::cerr<<"process number not match with arguments '-p' !!!\n";
		return 1;
	}
	/*if(nthreads != 1){
		std::cerr<<"require one thread !!!\n";
		return 1;
	}*/

	if(mtype==0)
		memtype = MemType::pageable;
	else if(mtype==1)
		memtype = MemType::pinned;
	else if(mtype ==2)
		memtype = MemType::unified;
	else
		std::cerr<<"wrong memory type for -m !!!"<<std::endl;

	/*
	 * init cluster objests
	 */
	FTYPE *objects;
	int numObjs;
	int numCoords;
	Task<ClusterDataInit<FTYPE>> clusterInit(ProcList(0));
	clusterInit.run(isBinaryFile, filename, &objects, &numObjs, &numCoords);
	clusterInit.wait();

	/*
	 * do cluster
	 */
	FTYPE *clusters = new FTYPE[numClusters*numCoords];
	/* Pick first numClusters elements of objects[] as initial cluster centers */
	for (int i=0; i < numClusters; i++)
		for (int j=0; j < numCoords; j++)
			clusters[i*numCoords + j] = objects[i*numCoords + j];

	double runtime_m[8][5];
	int loopcounters;
	Task<kmeansMGPU<FTYPE>> kmeans(ProcList(nthreads, 0), TaskType::gpu_task);
	kmeans.init(objects, clusters, numObjs, numCoords, numClusters);
	kmeans.run(runtime_m, threshold, &loopcounters, memtype);
	kmeans.wait();
	double runtime[5]={0,0,0,0,0};
	for(int i=0; i<nthreads; i++)
		for(int j=0; j<5; j++)
			runtime[j]+= runtime_m[i][j];
	for(int j=0; j<5; j++)
		runtime[j] /= nthreads;

	/*
	 * output
	 */
	if(outfile != NULL){
		Task<Output<FTYPE>> fileout(ProcList(0));
		fileout.run(outfile, clusters, numClusters, numCoords);
		fileout.wait();
	}

	if(objects)
		free(objects);
	if(clusters)
		free(clusters);

	std::cout<<"Test complete !!!"<<std::endl;
	if(printTime){
		printf("\n---- kMeans Clustering ----\n");
		printf("Input file:     %s\n", filename);
		printf("numObjs       = %d\n", numObjs);
		printf("numCoords     = %d\n", numCoords);
		printf("numClusters   = %d\n", numClusters);
		printf("threshold     = %.4f\n", threshold);
		printf("Iterations     	   = %d\n", loopcounters);
		printf("MemType: 		=%d\n", mtype);

		//printf("I/O time           = %10.4f sec\n", io_timing);
		printf("copyin time        = %10.4f sec\n", runtime[2]);
		printf("copyout time       = %10.4f sec\n", runtime[3]);
		printf("gpu kernel time    = %10.4f sec\n", runtime[1]);
		printf("host compute time  = %10.4f sec\n", runtime[4]);
		//clustering_timing = copyinTime + copyoutTime + kernelTime + hostCompTime;
		printf("Total time = %10.4f sec\n", runtime[0]);
	}

	for(int i=0; i<5; i++)
		runtime[i] *= 1000;
	print_time(5, runtime);

	return(0);

}








