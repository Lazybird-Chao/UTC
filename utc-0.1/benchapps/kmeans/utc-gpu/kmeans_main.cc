/*
 * kmeans_main.cc
 *
 *  Created on: Oct 13, 2017
 *      Author: chaoliu
 */

#include <iostream>
#include <iomanip>

#include "../../common/helper_getopt.h"
#include "../../common/helper_printtime.h"
#include "Utc.h"
#include "UtcGpu.h"
using namespace iUtc;

#include "task.h"
#include "./gpu/kmeans_task.h"

#define MAX_THREADS 64
#define MAX_TIMER 9

#define FTYPE float

int main(int argc, char **argv){
	bool printTime = false;
	int     isBinaryFile = 0;
	char    *filename = NULL;
	char    *outfile = NULL;
	FTYPE 	threshold = 0.01;
	int		maxIterations = 20;
	int 	numClusters = 1;

	int	nthreads = 1;
	int nprocess = 1;

	/* initialize UTC context */
	UtcContext &ctx = UtcContext::getContext(argc, argv);
	if(ctx.getProcRank() == 0)
		std::cout<<"UTC context initialized !\n";
	/* Parse command line options */
	int     opt;
	extern char   *optarg;
	extern int     optind;
	while ( (opt=getopt(argc,argv,"l:a:o:i:n:bvt:p:"))!= EOF) {
		switch (opt) {
			case 'v': printTime = true;
					  break;
			case 't': nthreads=atoi(optarg);
				  break;
			case 'p': nprocess = atoi(optarg);
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
			case 'l': maxIterations = atoi(optarg);
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
	if(nthreads != 1){
		std::cerr<<"only run one thread each node for task!!!\n";
		return 1;
	}

	/*
	 * init cluster objests
	 */
	FTYPE *objects = nullptr;
	int numObjs;
	int numCoords;
	Task<ClusterDataInit<FTYPE>> clusterInit(ProcList(0));
	clusterInit.run(isBinaryFile, filename, &objects, &numObjs, &numCoords);
	clusterInit.wait();

	/*
	 * do cluster
	 */
	FTYPE *clusters = nullptr;
	if(myproc == 0){
		clusters = new FTYPE[numClusters*numCoords];
		/* Pick first numClusters elements of objects[] as initial cluster centers */
		for (int i=0; i < numClusters; i++)
			for (int j=0; j < numCoords; j++)
				clusters[i*numCoords + j] = objects[i*numCoords + j];
	}
	double runtime_m[MAX_THREADS][MAX_TIMER];
	int loopcounters;
	ProcList plist;
	for(int i = 0; i < nprocess; i++)
		for(int j = 0; j<nthreads; j++)
			plist.push_back(i);
	Task<kmeansWorker<FTYPE>> kmeans(plist, TaskType::gpu_task);
	kmeans.init(objects, clusters, numObjs, numCoords, numClusters);
	kmeans.run(runtime_m, threshold, maxIterations,  &loopcounters);
	kmeans.wait();

	/*
	 * output
	 */
	if(outfile != NULL){
		Task<Output<FTYPE>> fileout(ProcList(0));
		fileout.run(outfile, clusters, numClusters, numCoords);
		fileout.wait();
	}

	if(myproc == 0){
		free(objects);
		free(clusters);
		double runtime[MAX_TIMER]={0,0,0,0,0,0,0,0,0};
		for(int i=0; i<nthreads; i++)
			for(int j=0; j<MAX_TIMER; j++)
				runtime[j]+= runtime_m[i][j];
		for(int j=0; j<MAX_TIMER; j++)
			runtime[j] /= nthreads;
		std::cout<<"Test complete !!!"<<std::endl;
		if(printTime){
			printf("\n---- kMeans Clustering ----\n");
			printf("Input file:     %s\n", filename);
			printf("numObjs       = %d\n", numObjs);
			printf("numCoords     = %d\n", numCoords);
			printf("numClusters   = %d\n", numClusters);
			printf("threshold     = %.4f\n", threshold);
			printf("Iterations     	   = %d\n", loopcounters);

			printf("total time        = %10.4f sec\n", runtime[0]);
			printf("compute time       = %10.4f sec\n", runtime[1]);
			printf("kernel time       = %10.4f sec\n", runtime[2]);
			printf("comm time    = %10.4f sec\n", runtime[3]);
			printf("copy time    = %10.4f sec\n", runtime[4]);
		}
		for(int i=0; i<MAX_TIMER; i++)
			runtime[i] *= 1000;
		print_time(5, runtime);
	}

	return 0;

}


