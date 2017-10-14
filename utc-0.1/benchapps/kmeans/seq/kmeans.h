/*
 * kmeans_compute.h
 *
 */

#ifndef BENCHAPPS_KMEANS_SEQ_KMEANS_H_
#define BENCHAPPS_KMEANS_SEQ_KMEANS_H_

#include <assert.h>

#define FTYPE float

FTYPE** kmeans(FTYPE**, int, int, int, FTYPE, int*, int , int*);

FTYPE** file_read(int, char*, int*, int*);




#endif /* BENCHAPPS_KMEANS_SEQ_KMEANS_H_ */
