/*
 * kmeans_main.cc
 *
 * The sequential k-means clustering program
 *
 * usage:
 * 		compile with the Makefile
 * 		run as: ./a.out -v -b -i file1 -o file2 -a 0.001 -n 10
 * 			-v: print time info
 * 			-b: inputfile is binary format
 * 			-i: inputfile path
 * 			-o: outpufile path
 * 			-a: threshold value for convergence
 * 			-n: number of clusters to class
 *
 */

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include "mpi.h"

#include "../../common/helper_getopt.h"
#include "../../common/helper_timer.h"
#include "../../common/helper_printtime.h"

#include "kmeans.h"


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

/*---< main() >-------------------------------------------------------------*/
int main(int argc, char **argv) {

	int numClusters, numCoords, numObjs;
    int     i, j;
    int     isBinaryFile;
    int    *membership;    /* [numObjs] */
    char   *filename, *outfile;
    FTYPE **objects = nullptr;       /* [numObjs][numCoords] data objects */
    FTYPE **clusters;      /* [numClusters][numCoords] cluster center */
    FTYPE   threshold;
    double  io_timing, clustering_timing;

    /* some default values */
    numClusters       = 1;		/* Amount of cluster centers */
    threshold         = 0.01; 	/* Percentage of objects that need to change membership for the clusting to continue */
    isBinaryFile      = 0;		/* 0 if the input file is in ASCII format, 1 for binary format */
    filename          = NULL;	/* Name of the input file */
    outfile           = NULL;
    bool printTime 	  = false;
    int maxIterations = 20;
    int nprocess = 1;

	/* Parse command line options */
    int     opt;
	extern char   *optarg;
	extern int     optind;
    while ( (opt=getopt(argc,argv,"p:l:a:o:i:n:bv"))!= EOF) {
        switch (opt) {
        	case 'v': printTime = true;
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
            case 'p': nprocess = atoi(optarg);
            		  break;
            case 'h': usage(argv[0]);
					  break;
            default: usage(argv[0]);
                      break;
        }
    }

    if (filename == NULL) usage(argv[0]);

    MPI_Init(&argc, &argv);
    int procs;
    int myproc;
    MPI_Comm_rank(MPI_COMM_WORLD, &myproc);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);
    if(nprocess != procs){
		std::cerr<<"process number not match with arguments '-p' !!!\n";
		return 1;
	}


    double t1, t2;
    if(myproc == 0){
    	std::cout<<"Reading objecs file"<<std::endl;
		t1 = getTime();
		/* Read input data points from given input file */
		objects = file_read(isBinaryFile, filename, &numObjs, &numCoords);
		assert(objects != NULL);
    }
	t2 = getTime();
	io_timing        	= t2 - t1;

	/*
	 * bcast number info
	 */
	MPI_Bcast(&numObjs, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&numCoords, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&numClusters, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if(objects == nullptr){
		objects = create_array_2d<FTYPE>(numObjs, numCoords);
	}

	int numLocalObjs = numObjs / nprocess;
	int localObjStartIndex = numLocalObjs * myproc;

	int *localMembership = new int[numLocalObjs];

	if(myproc == 0)
		std::cout<<"Start clustering..."<<std::endl;
	/* Launch the core computation algorithm */
	int iters = 0;
	t1 = getTime();
	double runtime[3];
    clusters = kmeans(objects, numCoords, numObjs,
                          numClusters, threshold, localMembership, maxIterations, &iters,
						  numLocalObjs, localObjStartIndex,
						  runtime);
    t2 = getTime();
    clustering_timing = t2 - t1;

    /* Memory cleanup */
    if(myproc == 0){
		free(objects[0]);
		free(objects);
    }
    free(localMembership);

    t1 = getTime();
    if(outfile != NULL && myproc == 0) {
        int l;
        FILE* fp = fopen(outfile, "w");
        for(j = 0; j < numClusters; j++) {
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

    double avg_runtime[3];
	MPI_Reduce(runtime, avg_runtime, 3, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    /* Print performance numbers on stdout */
    if(myproc == 0){
    	for(int i = 0; i< 3; i++)
			avg_runtime[i] /= nprocess;
		if(printTime){
			printf("\n---- kMeans Clustering ----\n");
			printf("Input file:     %s\n", filename);
			printf("numObjs       = %d\n", numObjs);
			printf("numCoords     = %d\n", numCoords);
			printf("numClusters   = %d\n", numClusters);
			printf("threshold     = %.4f\n", threshold);

			printf("Iterations         = %d\n", iters);
			printf("I/O time           = %10.4f sec\n", io_timing);
			printf("Computation timing = %10.4f sec\n", clustering_timing);
			printf("total time           = %10.4f sec\n", avg_runtime[0]);
			printf("compute time           = %10.4f sec\n", avg_runtime[1]);
			printf("comm time           = %10.4f sec\n", avg_runtime[2]);

		}
		for(int i = 0; i< 3; i++)
			avg_runtime[i] *= 1000;
		print_time(3, avg_runtime);
    }

    MPI_Finalize();
    return(0);
}
