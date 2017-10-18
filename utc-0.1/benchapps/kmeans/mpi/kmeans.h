/*
 * kmeans_compute.h
 *
 */

#ifndef BENCHAPPS_KMEANS_SEQ_KMEANS_H_
#define BENCHAPPS_KMEANS_SEQ_KMEANS_H_

#include <assert.h>

#define FTYPE float
#define MPI_FTYPE MPI_FLOAT

FTYPE** kmeans(FTYPE**, int, int, int, FTYPE, int*, int , int*,
		int, int, double*);

FTYPE** file_read(int, char*, int*, int*);

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


#endif /* BENCHAPPS_KMEANS_SEQ_KMEANS_H_ */
