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

#include "../../common/helper_getopt.h"
#include "../../common/helper_timer.h"

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
    bool printTime 	  = false;

	/* Parse command line options */
    int     opt;
	extern char   *optarg;
	extern int     optind;
    while ( (opt=getopt(argc,argv,"a:o:i:n:bv"))!= EOF) {
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
            case 'h': usage(argv[0]);
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
    objects = file_read(isBinaryFile, filename, &numObjs, &numCoords);
    assert(objects != NULL);

	t2 = getTime();
	io_timing        	= t2 - t1;


    membership = (int*) malloc(numObjs * sizeof(int));
	assert(membership != NULL);

	std::cout<<"Start clustering..."<<std::endl;
	/* Launch the core computation algorithm */
	int iters = 0;
	t1 = getTime();
    clusters = kmeans(objects, numCoords, numObjs,
                          numClusters, threshold, membership, &iters);
    t2 = getTime();
    clustering_timing = t2 - t1;

    /* Memory cleanup */
    free(objects[0]);
	free(objects);
    free(membership);

    t1 = getTime();
    if(outfile != NULL) {
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

    /* Print performance numbers on stdout */
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
    }
    return(0);
}
