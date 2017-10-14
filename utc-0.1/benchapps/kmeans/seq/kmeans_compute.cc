/*
 * kmeans_compute.cc
 *
 */

#include <cstdlib>
#include <cstdio>
#include <math.h>
#include <iostream>
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


/*
*	Function: create_array_2d
*	--------------------------
*	Allocates memory for a 2-dim array as needed for the algorithm.
*/
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
				   int	  *iters)        	/* out */
{
    int      i, j, k, index, loop = 0;
    int     *newClusterSize; /* [numClusters]: no. objects assigned in each
                                new cluster */
    FTYPE  **clusters;       /* out: [numClusters][numCoords] */
    FTYPE  **newClusters;    /* [numClusters][numCoords] */


	/* === MEMORY SETUP === */

	/* [numClusters] clusters of [numCoords] coordinates each */
	clusters = create_array_2d<FTYPE>(numClusters, numCoords);

    /* Pick first numClusters elements of objects[] as initial cluster centers */
    for (i=0; i < numClusters; i++)
        for (j=0; j < numCoords; j++)
            clusters[i][j] = objects[i][j];

    /* Initialize membership, no object belongs to any cluster yet */
    for (i = 0; i < numObjs; i++)
		membership[i] = -1;

	/* newClusterSize holds information on the count of members in each cluster */
    newClusterSize = (int*)calloc(numClusters, sizeof(int));
    assert(newClusterSize != NULL);

	/* newClusters holds the coordinates of the freshly created clusters */
	newClusters = create_array_2d<FTYPE>(numClusters, numCoords);

	/* === COMPUTATIONAL PHASE === */
    do {
		delta = 0.0;

        cluster(numClusters, numCoords, numObjs, membership, objects, newClusterSize, newClusters, clusters);

		/* Average the sum and replace old cluster centers with newClusters */
		for (i = 0; i < numClusters; i++) {
			for (j = 0; j < numCoords; j++) {
				clusters[i][j] = newClusters[i][j];
				if (newClusterSize[i] > 1)
					clusters[i][j] = newClusters[i][j] / newClusterSize[i];
				newClusters[i][j] = 0.0;   /* set back to 0 */
			}
			newClusterSize[i] = 0;   /* set back to 0 */
		}
		//std::cout<<delta<<std::endl;
		delta /= numObjs;
    } while (loop++ < maxIterations && delta > threshold);
    *iters = loop;

    free(newClusters[0]);
    free(newClusters);
    free(newClusterSize);
    return clusters;
}







