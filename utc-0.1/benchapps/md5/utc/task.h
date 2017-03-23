/*
 * task.h
 *
 *  Created on: Mar 22, 2017
 *      Author: chao
 */

#ifndef TASK_H_
#define TASK_H_

#include "md5.h"
#include "Utc.h"



class RandomInput : public UserTaskBase{
public:
	void runImpl(config_t *configArgs){
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
			return 1;
		// generate random data
		srand(datasets[index].rseed);
		// for cuda memory coalease, we store one buffer in a colum,
		// not a row
		for(int i=0; i<configArgs->numinputs; i++){
			uint8_t *p = &(configArgs->inputs[i]);
			for(int j = 0; j<configArgs->size; j++){
				p[j*configArgs->numinputs] = rand() % 255;
			}
		}
	}
};

class Output : public UserTaskBase{
public:
	void runImpl(config_t *args){
		char outname[] = "output.txt";
		char buffer[64];
		int offset = 0;
		FILE* fp;

		fp = fopen(outname, "w");

		for(int i = 0; i < args->numinputs; i++) {
			sprintf(buffer, "Buffer %d has checksum ", i);
			fwrite(buffer, sizeof(char), strlen(buffer)+1, fp);
			// the out is also stored in column order
			for(int j = 0; j < DIGEST_SIZE*2; j+=2) {
				sprintf(buffer+j, "%x", args->out[j/2*args->numinputs+i] & 0xf);
				sprintf(buffer+j+1, "%x", args->out[j/2*args->numinputs+i] & 0xf0);
			}
			buffer[32] = '\0';
			fwrite(buffer, sizeof(char), 32, fp);
			fputc('\n', fp);
		}

		fclose(fp);
	}
};



#endif /* TASK_H_ */
