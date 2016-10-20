/*
 * shmem_kmeans.c
 *
 *  Created on: Oct 19, 2016
 *      Author: chao
 */

/* user included file*/
#include "../helper_getopt.h"

#include <shmem.h>
#include <mpi.h>

/* other standard header file */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
extern int _debug;
int _debug;

long pSync[_SHMEM_REDUCE_SYNC_SIZE];
float pWrk[_SHMEM_REDUCE_MIN_WRKDATA_SIZE];
int pWrk2[_SHMEM_REDUCE_MIN_WRKDATA_SIZE];
int rank;
int nproc;

float** file_read(int, char*, int*, int*);
int     file_write(char*, int, int, int, float**, int*, int);

int read_n_objects(int, char*, int, int, float**);
float** mpi_read(int, char*, int*, int*, MPI_Comm);
int     mpi_write(int, char*, int, int, int, float**, int*, int, MPI_Comm,
                  int verbose);

/*---< usage() >------------------------------------------------------------*/
static void usage(char *argv0, float threshold) {
    char *help =
"Usage: %s [switches] -i filename -n num_clusters\n"
"       -i filename    : file containing data to be clustered\n"
"       -c centers     : file containing initial centers. default: filename\n"
"       -b             : input file is in binary format (default no)\n"
"       -r             : output file in binary format (default no)\n"
"       -n num_clusters: number of clusters (K must > 1)\n"
"       -t threshold   : threshold value (default %.4f)\n"
"       -o             : output timing results (default no)\n"
"       -v var_name    : using PnetCDF for file input and output and var_name\n"
"                        is variable name in the netCDF file to be clustered\n"
"       -k var_name    : name of variable in the netCDF to be used as the\n"
"                        initial cluster centers. If skipped, the variable\n"
"                        name from the option \"-v\" is used\n"
"       -q             : quiet mode\n"
"       -d             : enable debug mode\n"
"       -h             : print this help information\n";
    fprintf(stderr, help, argv0, threshold);
}

/*----< euclid_dist_2() >----------------------------------------------------*/
/* square of Euclid distance between two multi-dimensional points            */
__inline static
float euclid_dist_2(int    numdims,  /* no. dimensions */
                    float *coord1,   /* [numdims] */
                    float *coord2)   /* [numdims] */
{
    int i;
    float ans=0.0;

    for (i=0; i<numdims; i++)
        ans += (coord1[i]-coord2[i]) * (coord1[i]-coord2[i]);

    return(ans);
}

/*----< find_nearest_cluster() >---------------------------------------------*/
__inline static
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

/*----< osh_kmeans() >-------------------------------------------------------*/
int osh_kmeans(float    **objects,     /* in: [numObjs][numCoords] */
               int        numCoords,   /* no. coordinates */
               int        numObjs,     /* no. objects */
			   int 		  numTotalObjs,
               int        numClusters, /* no. clusters */
               float      threshold,   /* % objects change membership */
               int       *membership,  /* out: [numObjs] */
               float    **clusters,    /* out: [numClusters][numCoords] */
                int       *loops,
				double     *runtime)
{
	int      i, j, rank, index, loop=0, total_numObjs;
	int     *newClusterSize; /* [numClusters]: no. objects assigned in each
								new cluster */
	int     *clusterSize;    /* [numClusters]: temp buffer for Allreduce */
	          /* % of objects change their clusters */
	float    *delta_tmp = (float*)shmem_malloc(sizeof(float));
	float	 *delta = (float*)shmem_malloc(sizeof(float));
	float  **newClusters;    /* [numClusters][numCoords] */

	/* initialize membership[] */
	for (i=0; i<numObjs; i++) membership[i] = -1;

	/* need to initialize newClusterSize and newClusters[0] to all 0 */
	newClusterSize = (int*) shmem_malloc(numClusters*sizeof(int));
	assert(newClusterSize != NULL);
	for(int i=0; i<numClusters; i++)
		newClusterSize[i]=0;
	clusterSize    = (int*) shmem_malloc(numClusters*sizeof(int));
	assert(clusterSize != NULL);
	for(int i=0; i<numClusters; i++)
		clusterSize[i]=0;

	newClusters    = (float**) shmem_malloc(numClusters *            sizeof(float*));
	assert(newClusters != NULL);
	newClusters[0] = (float*)  shmem_malloc(numClusters * numCoords*sizeof(float));
	assert(newClusters[0] != NULL);
	for(int i=0; i<numClusters*numCoords; i++)
		newClusters[0][i]=0;
	for (i=1; i<numClusters; i++)
		newClusters[i] = newClusters[i-1] + numCoords;

	double t1, t2;
	t1=t2=0;
	do{
		double curT = MPI_Wtime();
		*delta = 0.0;
		for(i =0; i<numObjs; i++){
			/* find the array index of nestest cluster center */
			index = find_nearest_cluster(numClusters, numCoords, objects[i],
										 clusters);
			/* if membership changes, increase delta by 1 */
			if (membership[i] != index) *delta += 1.0;

			/* assign the membership to object i */
			membership[i] = index;

			/* update new cluster centers : sum of objects located within */
			newClusterSize[index]++;
			for (j=0; j<numCoords; j++)
				newClusters[index][j] += objects[i][j];
		}
		t1+=MPI_Wtime()-curT;
		/* sum all data objects in newClusters */
		shmem_float_sum_to_all (clusters[0],newClusters[0], numClusters*numCoords, 0,
		                                0, nproc, pWrk, pSync);
		shmem_barrier_all();
		shmem_int_sum_to_all (clusterSize, newClusterSize, numClusters, 0,
				                                0, nproc, pWrk2, pSync);
		shmem_barrier_all();
		/* average the sum and replace old cluster centers with newClusters */
		for (i=0; i<numClusters; i++) {
			for (j=0; j<numCoords; j++) {
				if (clusterSize[i] > 1)
					clusters[i][j] /= clusterSize[i];
				newClusters[i][j] = 0.0;   /* set back to 0 */
			}
			newClusterSize[i] = 0;   /* set back to 0 */
		}

		shmem_float_sum_to_all (delta_tmp, delta, 1, 0,
						                    0, nproc, pWrk, pSync);
		*delta = *delta_tmp / total_numObjs;
		shmem_barrier_all();
		t2+=MPI_Wtime()-curT;
	}while(*delta > threshold && loop++ < 100);

	*loops = loop;
	runtime[0]=t1;
	runtime[1]=t2-t1;
	runtime[2]=t2;

	shmem_free(newClusters[0]);
	shmem_free(newClusters);
	shmem_free(newClusterSize);
	shmem_free(clusterSize);
	return 1;
}


/*---< main() >-------------------------------------------------------------*/
int main(int argc, char **argv) {
           int     opt;
    extern char   *optarg;
    extern int     optind;
           int     i, j;
           int     isInFileBinary, isOutFileBinary, do_pnetcdf;
           int     is_output_timing, is_print_usage, verbose;

           int     numClusters, numCoords, numObjs, totalNumObjs;
           int    *membership; /* [numObjs] */
           char   *filename, *centers_filename;
           char   *var_name, *centers_name;
           float **objects;    /* [numObjs][numCoords] data objects */
           float **clusters;   /* [numClusters][numCoords] cluster centers */
           float   threshold;
           double  timing, io_timing, clustering_timing;

           //int     rank, nproc, mpi_namelen;
           //char    mpi_name[MPI_MAX_PROCESSOR_NAME];
           int N=1;

          shmem_init();
          nproc = shmem_n_pes();
          rank = shmem_my_pe();
          /* some default values */
		  _debug           = 0;
		  verbose          = 1;
		  threshold        = 0.001;
		  numClusters      = 0;
		  isInFileBinary   = 0;
		  isOutFileBinary  = 0;
		  is_output_timing = 0;
		  is_print_usage   = 0;
		  filename         = NULL;
		  do_pnetcdf       = 0;
		  var_name         = NULL;
		  centers_filename = NULL;
		  centers_name     = NULL;

		  while ( (opt=getopt(argc,argv,"i:n:t:v:c:l:abdorhq"))!= EOF) {
			  switch (opt) {
				  case 'i': filename=optarg;
							break;
				  case 'c': centers_filename=optarg;
							break;
				  case 'b': isInFileBinary = 1;
							break;
				  case 'r': isOutFileBinary = 1;
							break;
				  case 't': threshold=atof(optarg);
							break;
				  case 'n': numClusters = atoi(optarg);
							break;
				  case 'o': is_output_timing = 1;
							break;
				  case 'v': do_pnetcdf = 1;
							var_name = optarg;
							break;
				  case 'k': centers_name = optarg;
							break;
				  case 'q': verbose = 0;
							break;
				  case 'd': _debug = 1;
							break;
				  case 'l': N = atoi(optarg);
						break;
				  case 'h':
				  default: is_print_usage = 1;
							break;
			  }
		  }
		  if (filename == 0 || numClusters <= 1 || is_print_usage == 1 ||
			  (do_pnetcdf && var_name == NULL)) {
			  if (rank == 0) usage(argv[0], threshold);
			  MPI_Finalize();
			  exit(1);
		  }
		  if (centers_filename == NULL) centers_filename = filename;
		  if (centers_name     == NULL) centers_name     = var_name;

		  //if (_debug) printf("Proc %d of %d running on %s\n", rank, nproc, mpi_name);

          for (i = 0; i < _SHMEM_REDUCE_SYNC_SIZE; i += 1)
			  pSync[i] = _SHMEM_SYNC_VALUE;

		  shmem_barrier_all ();

		  /* read data points from file ------------------------------------------*/
		  if (rank == 0)
			  printf("reading data points from file %s\n",filename);
		  objects = mpi_read(isInFileBinary, filename, &numObjs, &numCoords,
		                             MPI_COMM_WORLD);
		  /* get the total number of data points */
		  MPI_Allreduce(&numObjs, &totalNumObjs, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
		  if (totalNumObjs < numClusters) {
			  if (rank == 0)
				  printf("Error: number of clusters must be larger than the number of data points to be clustered.\n");
			  free(objects[0]);
			  free(objects);
			  MPI_Finalize();
			  return 1;
		  }

		  /* allocate a 2D space for clusters[] (coordinates of cluster centers)
			 this array should be the same across all processes                  */
		  clusters    = (float**) shmem_malloc(numClusters *             sizeof(float*));
		  assert(clusters != NULL);
		  clusters[0] = (float*)  shmem_malloc(numClusters * numCoords * sizeof(float));
		  assert(clusters[0] != NULL);
		  for (i=1; i<numClusters; i++)
			  clusters[i] = clusters[i-1] + numCoords;

		  float **clusters_ori    = (float**) malloc(numClusters *             sizeof(float*));
		  assert(clusters_ori != NULL);
		  clusters_ori[0] = (float*)  malloc(numClusters * numCoords * sizeof(float));
		  assert(clusters_ori[0] != NULL);
		  for (i=1; i<numClusters; i++)
			  clusters_ori[i] = clusters_ori[i-1] + numCoords;

		  if (rank == 0) {
			 if (numObjs < numClusters || centers_filename != filename) {
				 printf("reading initial %d centers from file %s\n", numClusters,
						centers_filename);
				 /* read the first numClusters data points from file */
				 read_n_objects(isInFileBinary, centers_filename, numClusters,
								numCoords, clusters);
			 }
			 else {
				 printf("selecting the first %d elements as initial centers\n",
						numClusters);
				 /* copy the first numClusters elements in feature[] */
				 for (i=0; i<numClusters; i++)
					 for (j=0; j<numCoords; j++)
						clusters[i][j] = objects[i][j];
			 }
		 }
		 MPI_Bcast(clusters[0], numClusters*numCoords, MPI_FLOAT, 0, MPI_COMM_WORLD);
		 for (i=0; i<numClusters; i++)
			for (j=0; j<numCoords; j++)
				clusters_ori[i][j] = clusters[i][j];

		 clustering_timing = MPI_Wtime();
		 /* membership: the cluster id for each data object */
		 membership = (int*) malloc(numObjs * sizeof(int));
		 assert(membership != NULL);

		 /* start the core computation -------------------------------------------*/
		 int loops=0;
		 double runtime[3];
		 double runtimeTotal[3];
		 for(int i=0;i<3;i++)
			runtimeTotal[i]=0;
		 for(int k=0; k<N; k++){
		 	for (i=0; i<numClusters; i++)
		 		for (j=0; j<numCoords; j++)
		 			clusters[i][j] = clusters_ori[i][j];
		     osh_kmeans(objects, numCoords, numObjs, totalNumObjs, numClusters, threshold, membership,
		                clusters, &loops, runtime);
		     for(int j=0; j<3; j++)
		     	runtimeTotal[j]+=runtime[j];
		     MPI_Barrier(MPI_COMM_WORLD);
		}
		 free(objects[0]);
		 free(objects);

		 timing            = MPI_Wtime();
		 clustering_timing = timing - clustering_timing;
		 if(rank==0){
				file_write(filename, numClusters, numObjs, numCoords, clusters, membership,
						   verbose);
			}
		 free(membership);
		 shmem_free(clusters[0]);
		 shmem_free(clusters);
		 free(clusters_ori[0]);
		 free(clusters_ori);

		 double maxtotaltime[3];
		 MPI_Reduce(runtimeTotal, maxtotaltime, 3, MPI_DOUBLE,
		                            MPI_MAX, 0, MPI_COMM_WORLD);
		 if (rank == 0) {
			 printf("\nPerforming **** Simple Kmeans  (MPI) ****\n");
			 printf("Num of processes = %d\n", nproc);
			 printf("Input file:        %s\n", filename);
			 printf("numObjs          = %d\n", totalNumObjs);
			 printf("numCoords        = %d\n", numCoords);
			 printf("numClusters      = %d\n", numClusters);
			 printf("threshold        = %.4f\n", threshold);

			 printf("loops        = %d\n", loops);
			 printf("average time: %10.4f   %10.4f   %10.4f \n", maxtotaltime[0]/N, maxtotaltime[1]/N, maxtotaltime[2]/N);

		 }

		 shmem_finalize();
		 return 0;
}








