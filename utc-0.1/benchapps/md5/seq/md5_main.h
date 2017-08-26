/*
 * md5_main.h
 *
 *      Author: Chao
 */

#ifndef BENCHAPPS_MD5_SEQ_MD5_MAIN_H_
#define BENCHAPPS_MD5_SEQ_MD5_MAIN_H_

#include <cstdint>

#define DIGEST_SIZE 16

typedef struct {
    int input_set;
    int iterations;
    long numinputs;
    long size;
    int outflag;
    uint8_t* inputs;
    uint8_t* out;
} config_t;

typedef struct {
	long numbufs;
	long bufsize;
	int rseed;
} dataSet_t;

dataSet_t datasets[] ={
		{1024*2, 4094, 0},
		{1024*32, 4096*2, 1},
		{1024*256, 4096*4, 2},

	{64, 64*8, 0},//0
	{64, 64*16, 0},
	{64, 64*32, 0},
	{64, 64*64, 0},

	{1024*64, 512, 2}, //4
	{1024*64, 1024, 2},
	{1024*64, 2048, 2},
	{1024*64, 4096, 2},

	{1024*128, 512, 3}, //8
	{1024*128, 1024, 3},
	{1024*128, 2048, 3},
	{1024*128, 4096, 3},

	{128, 1024*512, 4}, //12
	{128, 1024*1024, 4},
	{128, 1024*2048, 4},
	{128, 1024*4096, 4},

	{1024, 1024*512, 4}, //16
	{1024, 1024*1024, 4},
	{1024, 1024*2048, 4},
	{1024, 1024*4096, 4},
};

int initialize(config_t *configArgs);
int finalize(config_t *configArgs);
void run(config_t *configArgs);
void process(uint8_t *in, uint8_t *out, long bufsize);




#endif /* BENCHAPPS_MD5_SEQ_MD5_MAIN_H_ */
