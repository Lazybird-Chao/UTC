/*
 * md5.h
 *
 *  Created on: Mar 22, 2017
 *      Author: chao
 */

#ifndef MD5_H_
#define MD5_H_


#include <cstdint>

#define DIGEST_SIZE 16

typedef struct {
    int input_set;
    int iterations;
    int numinputs;
    int size;
    int outflag;
    uint8_t* inputs;
    uint8_t* out;
} config_t;

typedef struct {
	int numbufs;
	int bufsize;
	int rseed;
} dataSet_t;





#endif /* MD5_H_ */
