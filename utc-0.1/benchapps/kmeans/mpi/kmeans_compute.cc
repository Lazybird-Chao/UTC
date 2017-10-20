/*
 * kmeans_compute.cc
 *
 */

#include <cstdlib>
#include <cstdio>
#include <math.h>
#include <iostream>
#include <cstring>
#include "mpi.h"
#include "kmeans.h"


#define PREC 20 // max iteration times

FTYPE delta; /* Delta is a value between 0 and 1 describing the percentage of objects which changed cluster membership */

/*
*	Function: euclid_dist_2
*	-----------------------
*	Computes the square of the euclidean distance between two multi-dimensional points.
*/
__inline static FTYPE euclid_dist_2(int numdims, FTYPE *coord1, FTYPE *coord2) {
    int i;
    FTYPE ans=0.0;

    for (i=0; i<numdims; i++)
        ans += (coord1[i]-coord2[i]) * (coord1[i]-coord2[i]);

    return(ans);
}

/*
*	Function: find_nearest_cluster
*	------------------------------
*	Function determining the cluster center which is closest to the given object.
*	Returns the index of that cluster center.
*/
__inline static int find_nearest_cluster(int numClusters, int numCoords, FTYPE *object, FTYPE **clusters) {
    int   index, i;
    FTYPE dist, min_dist;

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
    return index;
}

/*
void cluster(int numClusters, int numCoords, int numObjs, int* membership,
		FTYPE** objects, int* newClusterSize, FTYPE** newClusters, FTYPE** clusters) {
    for(int i = 0; i < numObjs; i++) {
        int index = find_nearest_cluster(numClusters, numCoords, objects[i], clusters);

        if(membership[i] != index) {
            delta += 1.0;
        }

        membership[i] = index;

        newClusterSize[index]++;
        for(int j = 0; j < numCoords; j++)
            newClusters[index][j] += objects[i][j];
    }
}
*/


/*
 * Algorithm main function
 */
FTYPE** kmeans(
                   FTYPE **objects,           	/* in: [numObjs][numCoords] */
                   int     numCoords,         	/* no. coordinates */
                   int     numObjs,           	/* no. objects */
                   int     numClusters,       	/* no. clusters */
                   FTYPE   threshold,         	/* % objects change membership */
                   int    *membership,			/* membership of each object */
                   int    maxIterations,
				   int	  *iters,        	/* out */
				   int		numLocalObjs,
				   int		localObjStartIndex,
				   double	*runtime)
{
    int      i, j, k, index, loop = 0;
    int     *newClusterSize; /* [numClusters]: no. objects assigned in each
                                new cluster */
    FTYPE  **clusters;       /* out: [numClusters][numCoords] */
    FTYPE  **newClusters;    /* [numClusters][numCoords] */

    int procs;
	int myproc;
	MPI_Comm_rank(MPI_COMM_WORLD, &myproc);
	MPI_Comm_size(MPI_COMM_WORLD, &procs);

	/* === MEMORY SETUP === */

	/* [numClusters] clusters of [numCoords] coordinates each */
	clusters = create_array_2d<FTYPE>(numClusters, numCoords);

    /* Pick first numClusters elements of objects[] as initial cluster centers */
	if(myproc == 0){
		for (i=0; i < numClusters; i++)
			for (j=0; j < numCoords; j++)
				clusters[i][j] = objects[i][j];
	}

    /* Initialize membership, no object belongs to any cluster yet */
    for (i = 0; i < numLocalObjs; i++)
		membership[i] = -1;

	/* newClusterSize holds information on the count of members in each cluster */
    newClusterSize = (int*)calloc(numClusters, sizeof(int));
    assert(newClusterSize != NULL);

    int* totalClusterSize = new int[numClusters];
    FTYPE totalDelta;

	/* newClusters holds the coordinates of the freshly created clusters */
	newClusters = create_array_2d<FTYPE>(numClusters, numCoords);
	MPI_Barrier(MPI_COMM_WORLD);

	double t0, t1;
	double totaltime = 0;
	double computetime = 0;
	double commtime = 0;
	t0 = MPI_Wtime();
	/*
	 * bcast obj info and init cluster info
	 */
	t1 = MPI_Wtime();
	MPI_Bcast(objects[0], numObjs*numCoords, MPI_FTYPE, 0, MPI_COMM_WORLD);
	MPI_Bcast(clusters[0], numClusters*numCoords, MPI_FTYPE, 0, MPI_COMM_WORLD);
	commtime += MPI_Wtime() - t1;
	//std::cout<<numClusters<<" "<<numObjs<<" "<<numCoords<<" "<<localObjStartIndex<<std::endl;
	/* === COMPUTATIONAL PHASE === */
    do {
		delta = 0.0;
		t1 = MPI_Wtime();
		memset(newClusters[0], 0, numClusters*numCoords*sizeof(FTYPE));
		memset(newClusterSize, 0, numClusters*sizeof(int));
		for(int i = 0; i < numLocalObjs; i++) {
			int index = find_nearest_cluster(numClusters, numCoords,
					objects[i+localObjStartIndex], clusters);

			if(membership[i] != index) {
				delta += 1.0;
			}

			membership[i] = index;

			newClusterSize[index]++;
			for(int j = 0; j < numCoords; j++)
				newClusters[index][j] += objects[i+localObjStartIndex][j];
		}
		computetime += MPI_Wtime() - t1;
		//std::cout<<myproc<<" "<<delta<<std::endl;
		/*
		 * gather new clusters from all nodes
		 */
		t1 = MPI_Wtime();
		MPI_Reduce(newClusters[0], clusters[0], numClusters*numCoords,
				MPI_FTYPE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(newClusterSize, totalClusterSize, numClusters,
						MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&delta, &totalDelta, 1,
								MPI_FTYPE, MPI_SUM, 0, MPI_COMM_WORLD);
		commtime += MPI_Wtime() - t1;

		if(myproc == 0){
			/* Average the sum and replace old cluster centers with newClusters */
			for (i = 0; i < numClusters; i++) {
				for (j = 0; j < numCoords; j++) {
					if (totalClusterSize[i] > 1)
						clusters[i][j] = clusters[i][j] / totalClusterSize[i];
					//newClusters[i][j] = 0.0;   /* set back to 0 */
				}
				//newClusterSize[i] = 0;   /* set back to 0 */
			}
			delta = totalDelta;
		}
		/*
		 * bcast new clusters and delta
		 */
		t1 = MPI_Wtime();
		MPI_Bcast(&delta, 1, MPI_FTYPE, 0, MPI_COMM_WORLD);
		MPI_Bcast(clusters[0], numClusters*numCoords, MPI_FTYPE, 0, MPI_COMM_WORLD);
		commtime += MPI_Wtime() - t1;
		//std::cout<<delta<<std::endl;
		delta /= numObjs;
    } while (loop++ < maxIterations && delta > threshold);
    MPI_Barrier(MPI_COMM_WORLD);
    totaltime = MPI_Wtime() - t0;
    runtime[0] = totaltime;
    runtime[1] = computetime;
    runtime[2] = commtime;

    *iters = loop;

    free(newClusters[0]);
    free(newClusters);
    free(newClusterSize);
    return clusters;
}







