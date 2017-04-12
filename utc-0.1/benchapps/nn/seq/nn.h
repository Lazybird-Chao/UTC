/*
 * kmeans_compute.h
 *
 */

#ifndef BENCHAPPS_KMEANS_SEQ_KMEANS_H_
#define BENCHAPPS_KMEANS_SEQ_KMEANS_H_

#include <assert.h>

#define FTYPE float


__inline static FTYPE euclid_dist_2(int numdims, FTYPE *coord1, FTYPE *coord2) {
    int i;
    FTYPE ans=0.0;

    for (i=0; i<numdims; i++)
        ans += (coord1[i]-coord2[i]) * (coord1[i]-coord2[i]);

    return(ans);
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


FTYPE** file_read(int, char*, int*, int*);




#endif /* BENCHAPPS_KMEANS_SEQ_KMEANS_H_ */
