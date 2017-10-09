/*
 * rotate_main.cc
 *
 *  Created on: Oct 6, 2017
 *      Author: chaoliu
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cerrno>

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

	char *infile_path=NULL;
	char *outfile_path=NULL;
	int angle=0;

	/* initialize UTC context */
	UtcContext &ctx = UtcContext::getContext(argc, argv);

	/* Parse command line options */
	int     opt;
	extern char   *optarg;
	extern int     optind;
	while ( (opt=getopt(argc,argv,"a:i:o:vt:p:"))!= EOF) {
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
			case 'a': angle = atoi(optarg);
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
	if(nprocess != 1){
		std::cerr<<"This test only run with one node !!!\n";
		return 1;
	}
	if(infile_path == NULL){
		std::cerr<<"Error, need the input image file."<<std::endl;
		return 1;
	}
	if(angle ==0 || angle ==360){
		std::cout<<"No need to rotate, same as original image."<<std::endl;
		return 0;
	}

	/*
	 * create image object from the file
	 */
	Image srcImg;
	Task<ImageCreate> readImg(ProcList(0));
	readImg.run(&srcImg, infile_path);
	readImg.wait();

	/*
	 * do rotation
	 */
	Image dstImg;
	double runtime_m[MAX_THREADS][1];
	Task<RotateWorker> rotate(ProcList(0), TaskType::gpu_task);
	rotate.init(&srcImg, &dstImg, angle);
	rotate.run(runtime_m);
	rotate.wait();


	/*
	 * out put the image
	 */
	if(outfile_path != NULL){
		Task<ImageOut> writeImg(ProcList(0));
		writeImg.run(&dstImg, outfile_path);
		writeImg.wait();
	}

	srcImg.clean();
	dstImg.clean();

	std::cout<<"Test complete !!!"<<std::endl;
	if(printTime){
		double runtime = 0;
		for(int i = 0; i<nthreads; i++)
			runtime += runtime_m[i][0];
		runtime /= nthreads;
		std::cout<<"\tInput image size: "<<srcImg.getWidth()<<" X "<<srcImg.getHeight()<<std::endl;
		std::cout<<"\tOutput image size: "<<dstImg.getWidth()<<" X "<<dstImg.getHeight()<<std::endl;
		std::cout<<"\tTime info: "<<std::endl;
		std::cout<<"\t\ttotal run time: "<<std::fixed<<std::setprecision(4)<<1000*runtime<<"(ms)"<<std::endl;

		runtime *= 1000;
		print_time(4, &runtime);
	}

	return 0;

}

