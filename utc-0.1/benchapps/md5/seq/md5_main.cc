/*
 * md5_main.cc
 *
 *      Author: Chao
 *
 * Sequential version of MD5 program.
 *
 * usage:
 * 		compile with the Makefile
 * 		run as: ./a.out -v -i 1 -c 1 -o 0
 * 			-v: print time info
 * 			-i: select which data sets will be used for run
 * 			-c: number of iterations
 * 			-o: if write result to file
 *
 */

#include "md5_main.h"
#include "md5_compute.h"

#include "../../common/helper_getopt.h"
#include "../../common/helper_timer.h"

#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cstdio>
#include <cstring>

int main(int argc, char* argv[]){
	bool printTime = false;
	config_t configArgs;
	configArgs.input_set = 0;
	configArgs.iterations = 1;
	configArgs.outflag = 0;

	/*
	 * parse arguments
	 */
	int opt;
	extern char* optarg;
	extern int optind;
	opt=getopt(argc, argv, "vi:c:o");
	while(opt!=EOF){
		switch(opt){
		case 'v':
			printTime = true;
			break;
		case 'i':
			configArgs.input_set = atoi(optarg);
			break;
		case 'c':
			configArgs.iterations = atoi(optarg);
			break;
		case 'o':
			configArgs.outflag = 1;
			break;
		case '?':
			break;
		default:
			break;
		}
		opt=getopt(argc, argv, "vi:c:o");
	}

	// parameter initalization
	std::cout<<"Generating random input data set ..."<<std::endl;
	if(initialize(&configArgs)){
		std::cerr<<"Initialization Error !!!"<<std::endl;
		return 1;
	}

	// do md5 processing
	std::cout<<"Start MD5 processing ..."<<std::endl;
	double t1, t2;
	t1 = getTime();
	run(&configArgs);
	t2 = getTime();
	double runtime = t2-t1;

	// write result
	if(finalize(&configArgs)){
		std::cerr<<"Finalization Error !!!"<<std::endl;
		return 1;
	}

	std::cout<<"Test complete !!!"<<std::endl;
	if(printTime){
		std::cout<<"\tprocess data info:"<<std::endl;
		std::cout<<"\t\tnumber buffs:"<<datasets[configArgs.input_set].numbufs<<std::endl;
		std::cout<<"\t\tbuff size(Bytes):"<<datasets[configArgs.input_set].bufsize<<std::endl;
		std::cout<<"\ttime info:"<<std::fixed<<std::setprecision(4)<<1000*runtime<<"(ms)"<<std::endl;
	}

	return 0;


}

int initialize(config_t *configArgs){
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
	for(int i=0; i<configArgs->numinputs; i++){
		uint8_t *p = &(configArgs->inputs[i*configArgs->size]);
		for(int j = 0; j<configArgs->size; j++)
			*p++ = rand() % 255;
	}
	return 0;
}

/*
*   Function: process
*   -----------------
*   Processes one input buffer, delivering the digest into out.
*/
void process(uint8_t* in, uint8_t* out, long bufsize) {
    MD5_CTX context;
    uint8_t digest[16];

    MD5_Init(&context);
    MD5_Update(&context, in, bufsize);
    //MD5_CTX *ctx = &context;
    //std::cout<<ctx->a<<" "<<ctx->b<<" "<<ctx->c<<" "<<ctx->d<<std::endl;
    MD5_Final((unsigned char*)digest, &context);

    memcpy(out, digest, DIGEST_SIZE);
}


/*
*   Function: run
*   --------------------
*   Main benchmarking function. If called, processes buffers with MD5
*   until no more buffers available. The resulting message digests
*   are written into consecutive locations in the preallocated output
*   buffer.
*/
void run(config_t * args) {
	for(int i = 0; i < args->iterations; i++) {

		int buffers_to_process = args->numinputs;
		uint8_t* in = args->inputs;
		uint8_t* out = args->out;

		int j=0;
		while(j< buffers_to_process) {
			process(&in[j*args->size], out+j*DIGEST_SIZE, args->size);
			j++;
		}

	}
}

int finalize(config_t *args){
	char outname[] = "output.txt";
	char buffer[64];
	int offset = 0;
	FILE* fp;

	if(args->outflag) {
		fp = fopen(outname, "w");

		for(int i = 0; i < args->numinputs; i++) {
			sprintf(buffer, "Buffer %d has checksum ", i);
			fwrite(buffer, sizeof(char), strlen(buffer)+1, fp);
			for(int j = 0; j < DIGEST_SIZE*2; j+=2) {
				sprintf(buffer+j, "%x", args->out[DIGEST_SIZE*i+j/2] & 0xf);
				sprintf(buffer+j+1, "%x", args->out[DIGEST_SIZE*i+j/2] & 0xf0);
			}
			buffer[32] = '\0';
			fwrite(buffer, sizeof(char), 32, fp);
			fputc('\n', fp);
		}

		fclose(fp);
	}

	if(args->inputs) {
		free(args->inputs);
	}

	if(args->out)
		free(args->out);

	return 0;
}

















