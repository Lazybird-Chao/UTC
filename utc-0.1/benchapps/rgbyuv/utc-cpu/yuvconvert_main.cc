/*
 * yuvconvert_main.cc
 *
 *  Created on: Oct 9, 2017
 *      Author: Chao
 */

#include <cstring>
#include <iostream>
#include <cstdlib>
#include <iomanip>
#include <cuda_runtime.h>

#include "../../common/helper_getopt.h"
#include "../../common/helper_printtime.h"
#include "Utc.h"

#include "image.h"
#include "task.h"

#define MAX_THREADS 64

using namespace iUtc;

int main(int argc, char** argv){
	bool printTime = false;
	int nthreads=1;
	int nprocess=1;

	char *infile_path = NULL;
	char *outfile_path = NULL;
	int iterations=1;

	/* initialize UTC context */
	UtcContext &ctx = UtcContext::getContext(argc, argv);

	/* Parse command line options */
	int     opt;
	extern char   *optarg;
	extern int     optind;
	while ( (opt=getopt(argc,argv,"l:i:o:vt:p:"))!= EOF) {
		switch (opt) {
			case 'v': printTime = true;
					  break;
			case 't': nthreads=atoi(optarg);
					  break;
			case 'p': nprocess = atoi(optarg);
					  break;
			case 'i': infile_path=optarg;
					  break;
			case 'o': outfile_path = optarg;
					  break;
			case 'l': iterations = atoi(optarg);
					  break;
			case ':':
				std::cerr<<"Option -"<<(char)optopt<<" requires an operand\n"<<std::endl;
				break;
			case '?':
				std::cerr<<"Unrecognized option: -"<<(char)optopt<<std::endl;
				break;
			default:
					  break;
		}
	}
	int procs = ctx.numProcs();
	int myproc = ctx.getProcRank();
	if(nprocess != procs){
		std::cerr<<"process number not match with arguments '-p' !!!\n";
		return 1;
	}
	/*if(nthreads != 1){
		std::cerr<<"require one thread !!!\n";
		return 1;
	}*/

	if(infile_path == NULL){
		std::cerr<<"Error, need the input image file."<<std::endl;
		return 1;
	}


	/*
	 * create image object from the file
	 */
	Image srcImg;
	Task<ImageCreate> readImg(ProcList(0));
	readImg.run(&srcImg, infile_path);
	readImg.wait();

	//
	//srcImg.increaseHeightBy(1);


	/*
	 * do converter
	 */
	yuv_color_t dstImg;
	double runtime_m[MAX_THREADS][1];
	Task<YUVconvertWorker> yuvconvert(ProcList(nthreads, 0), TaskType::cpu_task);
	yuvconvert.init(&srcImg, &dstImg);
	yuvconvert.run(runtime_m, iterations);
	yuvconvert.wait();

	/*
	 * out put the image
	 */
	if(outfile_path != NULL){
		Task<ImageOut> writeImg(ProcList(0));
		writeImg.run(&dstImg, srcImg.getWidth(), srcImg.getHeight(), outfile_path);
		writeImg.wait();
	}
	srcImg.clean();
	free(dstImg.y);
	free(dstImg.u);
	free(dstImg.v);

	if(myproc == 0){
		std::cout<<"Test complete !!!"<<std::endl;
		double runtime = 0;
		for(int i=0; i<nthreads; i++)
			runtime += runtime_m[i][0];
		for(int j=0; j<1; j++)
			runtime /= nthreads;

		if(printTime){
			std::cout<<"\tInput image size: "<<srcImg.getWidth()<<" X "<<srcImg.getHeight()<<std::endl;
			std::cout<<"\tTime info: "<<std::endl;
			std::cout<<"\t\ttotal time: "<<std::fixed<<std::setprecision(4)<<1000*runtime<<"(ms)"<<std::endl;
		}

		runtime *= 1000;
		print_time(1, &runtime);
	}

	return 0;

}
