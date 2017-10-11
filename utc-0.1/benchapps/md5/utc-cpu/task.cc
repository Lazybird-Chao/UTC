/*
 * task.cc
 *
 *  Created on: Oct 10, 2017
 *      Author: chaoliu
 */

#include "task.h"
#include "md5.h"
#include <iostream>
#include <fstream>
#include <cstring>

using namespace iUtc;

dataSet_t datasets[] ={
	{128*1024, 16*1024, 0},
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

void RandomInput::runImpl(config_t *configArgs, const char* filename, bool isBinary){
	if(filename == nullptr){
		int index = configArgs->input_set;
		if(index < 0 || index >= sizeof(datasets)/sizeof(datasets[0])) {
			std::cout<<"Invalid input set choice, set to default 0"<<std::endl;
			index = 0;
		}

		configArgs->numinputs = datasets[index].numbufs;
		configArgs->size = datasets[index].bufsize;
		configArgs->inputs = (uint8_t*)calloc(configArgs->numinputs*configArgs->size, sizeof(uint8_t));
		configArgs->out = (uint8_t*)calloc(configArgs->numinputs, DIGEST_SIZE);
		if(configArgs->inputs ==NULL || configArgs->out==NULL)
			return;
		// generate random data
		srand(datasets[index].rseed);

		for(long i=0; i<configArgs->numinputs; i++){
			uint8_t *p = &(configArgs->inputs[i*configArgs->size]);
			int key = rand();
			for(long j = 0; j<configArgs->size; j++){
				p[j] = (key+j) % 255;
			}
		}
	}
	else{
		long numBuffs, buffSize;
		char *data;
		fromFile(data, numBuffs, buffSize, filename, isBinary);
		configArgs->numinputs = numBuffs;
		configArgs->size = buffSize;
		configArgs->inputs = (uint8_t*)data;
		configArgs->out = (uint8_t*)calloc(configArgs->numinputs, DIGEST_SIZE);
	}
	return;
}

void Output::runImpl(config_t *args){
	char outname[] = "output.txt";
	char buffer[64];
	int offset = 0;
	FILE* fp;

	fp = fopen(outname, "w");

	for(long i = 0; i < args->numinputs; i++) {
		sprintf(buffer, "Buffer %d has checksum ", i);
		fwrite(buffer, sizeof(char), strlen(buffer)+1, fp);
		// the out is also stored in column order
		for(long j = 0; j < DIGEST_SIZE*2; j+=2) {
			sprintf(buffer+j, "%x", args->out[j/2 + i*DIGEST_SIZE] & 0xf);
			sprintf(buffer+j+1, "%x", args->out[j/2 + i*DIGEST_SIZE] & 0xf0);
		}
		buffer[32] = '\0';
		fwrite(buffer, sizeof(char), 32, fp);
		fputc('\n', fp);
	}

	fclose(fp);
}


void toFile(char* data, long numBuffs, long buffSize, const char* filename, bool isBinary){
	std::ofstream outfile;
	if(isBinary){
		outfile.open(filename, std::ofstream::binary);
		outfile.write((char*)&numBuffs, sizeof(long));
		outfile.write((char*)&buffSize, sizeof(long));
		outfile.write(data, numBuffs*buffSize);
	}
	else{
		outfile.open(filename);
		outfile<<numBuffs<<" "<<buffSize<<std::endl;
		for(long i=0; i<numBuffs; i++){
			for(long j=0; j<buffSize; j++)
				outfile<<data[i*buffSize+j];
			outfile<<std::endl;
		}
	}
	return;
}

void fromFile(char* &data, long& numBuffs, long& buffSize, const char* filename, bool isBinary){
	std::cout<<"read input from file ...\n";
	std::ifstream infile;
	if(isBinary){
		infile.open(filename, std::ios::binary);
		infile.read((char*)&numBuffs, sizeof(long));
		infile.read((char*)&buffSize, sizeof(long));
		//std::cout<<numBuffs<<" "<<buffSize<<std::endl;
		data = new char[numBuffs*buffSize];
		infile.read(data, numBuffs*buffSize);
	}
	else{
		infile.open(filename);
		infile>>numBuffs;
		infile>>buffSize;
		data = new char[numBuffs*buffSize];
		for(long i=0; i<numBuffs; i++){
			for(long j=0; j<buffSize; j++)
				infile>>data[i*buffSize+j];
		}
	}
	return;
}

void increaseBy(int times, config_t *configArgs){
	if(times == 1)
		return;
	long numBuffs = configArgs->numinputs * times;
	char* data = new char[numBuffs * configArgs->size];
	for(long i=0; i<times; i++)
		memcpy(data+i*configArgs->numinputs * configArgs->size, configArgs->inputs, configArgs->numinputs * configArgs->size);
	configArgs->numinputs = numBuffs;
	free(configArgs->inputs);
	configArgs->inputs = (uint8_t*)data;
	free(configArgs->out);
	configArgs->out = (uint8_t*)calloc(configArgs->numinputs, DIGEST_SIZE);
	return;
}



