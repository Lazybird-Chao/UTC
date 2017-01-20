/*
 * kmeans_mian_v2.cu
 *
 *  Created on: Jan 19, 2017
 *      Author: chao
 */
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <assert.h>

#include "../../common/helper_getopt.h"
#include "../../common/helper_timer.h"
#include "../../common/helper_err.h"

#include "file_io.h"
#include "kmeans_kernel_v2.h"

#define FTYPE float
#define PREC 300 // max iteration times


/*
*	Function: usage
*	---------------
*	Prints information on how to call the program.
*/
static void usage(char *argv0) {
    char *help =
        "Usage: %s [switches] -i filename -n num_clusters [OPTIONS]\n"
        "       -i filename    : file containing data to be clustered\n"
        "       -b             : input file is in binary format (default no)\n"
        "       -n num_clusters: number of clusters (K must be > 1)\n"
        "       -o filename    : write output to file\n";
    fprintf(stderr, help, argv0);
    exit(-1);
}

template<typename T>
T** create_array_2d(int height, int width) {
	T** ptr;
	int i;
	ptr = (T**)calloc(height, sizeof(T*));
	assert(ptr != NULL);
	ptr[0] = (T*)calloc(width * height, sizeof(T));
	assert(ptr[0] != NULL);
	/* Assign pointers correctly */
	for(i = 1; i < height; i++)
		ptr[i] = ptr[i-1] + width;
	return ptr;
}


/*---< main() >-------------------------------------------------------------*/
int main(int argc, char **argv) {

	int numClusters, numCoords, numObjs;

    int     isBinaryFile;
    int    *membership;    /* [numObjs] */
    char   *filename, *outfile;
    FTYPE **objects;       /* [numObjs][numCoords] data objects */
    FTYPE **clusters;      /* [numClusters][numCoords] cluster center */
    FTYPE   threshold;
    double  io_timing, clustering_timing;

    /* some default values */
    numClusters       = 1;		/* Amount of cluster centers */
    threshold         = 0.001; 	/* Percentage of objects that need to change membership for the clusting to continue */
    isBinaryFile      = 0;		/* 0 if the input file is in ASCII format, 1 for binary format */
    filename          = NULL;	/* Name of the input file */
    outfile           = NULL;

	/* Parse command line options */
    int     opt;
	extern char   *optarg;
	extern int     optind;
    while ( (opt=getopt(argc,argv,"o:i:n:b"))!= EOF) {
        switch (opt) {
            case 'i': filename=optarg;
                      break;
            case 'b': isBinaryFile = 1;
                      break;
            case 'n': numClusters = atoi(optarg);
                      break;
            case 'h': usage(argv[0]);
                      break;
            case 'o': outfile = optarg;
                      break;
            default: usage(argv[0]);
                      break;
        }
    }

    if (filename == NULL) usage(argv[0]);


    double t1, t2;
    std::cout<<"Reading objecs file"<<std::endl;
    t1 = getTime();
    /* Read input data points from given input file */
    objects = file_read<FTYPE>(isBinaryFile, filename, &numObjs, &numCoords);
    assert(objects != NULL);
	t2 = getTime();
	io_timing        	= t2 - t1;


    membership = (int*) malloc(numObjs * sizeof(int));
    int *new_membership = (int*) malloc(numObjs * sizeof(int));
    clusters = create_array_2d<FTYPE>(numClusters, numCoords);
    FTYPE **new_clusters = create_array_2d<FTYPE>(numClusters, numCoords);
    int *new_clustersize = (int*)calloc(numClusters, sizeof(int));
    /* Pick first numClusters elements of objects[] as initial cluster centers */
	for (int i=0; i < numClusters; i++)
		for (int j=0; j < numCoords; j++)
			clusters[i][j] = objects[i][j];

	/* Initialize membership, no object belongs to any cluster yet */
	for (int i = 0; i < numObjs; i++)
		membership[i] = -1;

    /*
     * create gpu memory
     */
	cudaSetDevice(0);
    FTYPE *objects_d;
    int *membership_d;
    FTYPE *clusters_d;
    checkCudaErr(cudaMalloc(&objects_d, sizeof(FTYPE)*numObjs*numCoords));
    checkCudaErr(cudaMalloc(&membership_d, sizeof(int)*numObjs));
    checkCudaErr(cudaMalloc(&clusters_d, sizeof(FTYPE)*numClusters*numCoords));

    /*
     * copy data in
     */
    t1 = getTime();
    checkCudaErr(cudaMemcpy(objects_d, objects[0], sizeof(FTYPE)*numObjs*numCoords, cudaMemcpyHostToDevice));
    t2 = getTime();
    double copyinTime = t2- t1;

    /*
	 * copy in new clusters data
	 */
	t1 = getTime();
	checkCudaErr(cudaMemcpy(clusters_d, clusters[0], sizeof(FTYPE)*numClusters*numCoords, cudaMemcpyHostToDevice));
	t2 = getTime();
	copyinTime += t2-t1;

    /*
     * kernel computing
     */
	std::cout<<"Start clustering..."<<std::endl;
	double kernelTime =0;
	double copyoutTime = 0;
	double hostCompTime = 0;

	int batchPerThread = 16;
	int blocksize = 256;
	int gridsize = (numObjs + blocksize*batchPerThread -1)/(blocksize*batchPerThread);
	dim3 membership_kernel_block(blocksize, 1, 1);
	dim3 membership_kernel_grid(gridsize, 1, 1);
	FTYPE *new_clusters_reduce_d;
	int *new_clusters_size_reduce_d;
	int *new_clusters_size_d;
	int *change_count_reduce_d;
	int *change_count_reduce;
	checkCudaErr(cudaMalloc(&new_clusters_reduce_d, sizeof(FTYPE)*numClusters*numCoords*gridsize));
	checkCudaErr(cudaMalloc(&new_clusters_size_reduce_d, sizeof(int)*numClusters*gridsize));
	checkCudaErr(cudaMalloc(&new_clusters_size_d, sizeof(int)*numClusters));
	checkCudaErr(cudaMalloc(&change_count_reduce_d, sizeof(int)*gridsize));
	change_count_reduce = (int*)malloc(sizeof(int)*gridsize);

	dim3 new_clusters_size_kernel_block(blocksize, 1, 1);
	dim3 new_clusters_size_kernel_grid((numClusters+ blocksize-1)/blocksize, 1, 1);

	dim3 new_clusters_kernel_block(blocksize, 1, 1);
	int gridsize2 = (numClusters*numCoords + blocksize-1)/(blocksize);
	dim3 new_clusters_kernel_grid(gridsize2, 1, 1);


	int changedObjs =0;
	int loopcounters = 0;
	do{

		std::cout<<__LINE__<<std::endl;
		t1 = getTime();
		membership_kernel<<<membership_kernel_grid,
				membership_kernel_block>>>(objects_d, numCoords, numObjs, numClusters,
                          clusters_d, membership_d, batchPerThread,
                          new_clusters_reduce_d, new_clusters_size_reduce_d,
                          change_count_reduce_d);
		cudaDeviceSynchronize();
		std::cout<<__LINE__<<std::endl;
		new_clusters_size_kernel<<<new_clusters_size_kernel_grid,
					new_clusters_size_kernel_block>>>(new_clusters_size_reduce_d,
							new_clusters_size_d,
							numClusters, gridsize);
		cudaDeviceSynchronize();
		new_clusters_kernel<<<new_clusters_kernel_grid,
				new_clusters_kernel_block>>>(new_clusters_reduce_d,
						clusters_d,
						new_clusters_size_d,
						numClusters, numCoords, gridsize);
		cudaDeviceSynchronize();
		checkCudaErr(cudaGetLastError());
		t2 = getTime();
		kernelTime += t2-t1;
		std::cout<<__LINE__<<std::endl;
		/*
		 * copy out change obj counts
		 */
		t1 = getTime();
		checkCudaErr(cudaMemcpy(change_count_reduce, change_count_reduce_d, sizeof(int)*gridsize, cudaMemcpyDeviceToHost));
		t2 = getTime();
		copyoutTime += t2-t1;

		/*
		 * compute new clusters
		 */
		t1 = getTime();
		changedObjs = 0;
		for(int i=0; i<gridsize; i++)
			changedObjs += change_count_reduce[i];
		t2 = getTime();
		hostCompTime += t2-t1;

	}while(loopcounters++ < PREC && (FTYPE)changedObjs/numObjs > threshold );


    /* Memory cleanup */
    free(objects[0]);
	free(objects);
    free(membership);
    free(new_clusters[0]);
    free(new_clusters);
    free(new_membership);
    free(new_clustersize);
    cudaFree(objects_d);
    cudaFree(membership_d);
    cudaFree(clusters_d);

    cudaFree(new_clusters_reduce_d);
    cudaFree(new_clusters_size_reduce_d);
    cudaFree(new_clusters_size_d);
    cudaFree(change_count_reduce_d);
    free(change_count_reduce);


    t1 = getTime();
    if(outfile != NULL) {
        int l;
        FILE* fp = fopen(outfile, "w");
        for(int j = 0; j < numClusters; j++) {
            fprintf(fp, "Cluster %d: ", j);
            for(l = 0; l < numCoords; l++)
                fprintf(fp, "%f ", clusters[j][l]);
            fprintf(fp, "\n");
        }
        fclose(fp);
    }
    t2 = getTime();
    io_timing += t2 - t1;
    free(clusters[0]);
    free(clusters);

    /* Print performance numbers on stdout */

    printf("\n---- kMeans Clustering ----\n");
    printf("Input file:     %s\n", filename);
    printf("numObjs       = %d\n", numObjs);
    printf("numCoords     = %d\n", numCoords);
    printf("numClusters   = %d\n", numClusters);
    printf("threshold     = %.4f\n", threshold);

    printf("Iterations     	   = %d\n", loopcounters);
    printf("I/O time           = %10.4f sec\n", io_timing);
    printf("copyin time        = %10.4f sec\n", copyinTime);
    printf("copyout time       = %10.4f sec\n", copyoutTime);
    printf("gpu kernel time    = %10.4f sec\n", kernelTime);
    printf("host compute time  = %10.4f sec\n", hostCompTime);
    clustering_timing = copyinTime + copyoutTime + kernelTime + hostCompTime;
    printf("Computation timing = %10.4f sec\n", clustering_timing);

    return(0);
}



