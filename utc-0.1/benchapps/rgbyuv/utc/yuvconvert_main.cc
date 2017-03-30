/*
 * yuvconvert_main.cc
 *
 *  Created on: Mar 17, 2017
 *      Author: chao
 *
 * usage:
 * 		compile with the Makefile
 *		run as: ./a.out -v -i in.ppm -o out  -l 100
 *			-v: print time info
 *			-t: number of threads (not useful here)
 * 			-p: number of process (not useful here)
 * 			-m: gpu memory type (0:pageable, 1:pinned, 2:unified)
 *			-i: input image file
 *			-o: output image file path
 *			-l: iterations of the kernel
 */

#include <cstring>
#include <iostream>
#include <cstdlib>
#include <iomanip>
#include <cuda_runtime.h>

#include "../../common/helper_getopt.h"
#include "../../common/helper_err.h"
#include "../../common/helper_printtime.h"
#include "Utc.h"
#include "UtcGpu.h"

#include "image.h"
#include "task.h"
#include "./sgpu/yuv_task_sgpu.h"


using namespace iUtc;

int main(int argc, char** argv){
	bool printTime = false;
	int nthreads=1;
	int nprocess=1;

	MemType memtype = MemType::pageable;
	int mtype = 0;
	char *infile_path = NULL;
	char *outfile_path = NULL;
	int iterations=1;

	/* initialize UTC context */
	UtcContext &ctx = UtcContext::getContext(argc, argv);

	/* Parse command line options */
	int     opt;
	extern char   *optarg;
	extern int     optind;
	while ( (opt=getopt(argc,argv,"l:i:o:vt:p:m:"))!= EOF) {
		switch (opt) {
			case 'v': printTime = true;
					  break;
			case 't': nthreads=atoi(optarg);
					  break;
			case 'p': nprocess = atoi(optarg);
					  break;
			case 'm': mtype = atoi(optarg);
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
	if(nthreads != 1){
		std::cerr<<"require one thread !!!\n";
		return 1;
	}

	if(mtype==0)
		memtype = MemType::pageable;
	else if(mtype==1)
		memtype = MemType::pinned;
	else if(mtype ==2)
		memtype = MemType::unified;
	else
		std::cerr<<"wrong memory type for -m !!!"<<std::endl;

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

	/*
	 * do converter
	 */
	yuv_color_t dstImg;
	double runtime[4];
	Task<YUVconvertSGPU> yuvconvert(ProcList(0), TaskType::gpu_task);
	yuvconvert.init(&srcImg, &dstImg);
	yuvconvert.run(runtime, iterations, memtype);
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

	std::cout<<"Test complete !!!"<<std::endl;
	if(printTime){
		std::cout<<"\tInput image size: "<<srcImg.getWidth()<<" X "<<srcImg.getHeight()<<std::endl;
		std::cout<<"\tMemtype: "<<mtype<<std::endl;
		std::cout<<"\tTime info: "<<std::endl;
		std::cout<<"\t\tmemcpy in time: "<<std::fixed<<std::setprecision(4)<<1000*runtime[1]<<"(ms)"<<std::endl;
		std::cout<<"\t\tmemcpy out time: "<<std::fixed<<std::setprecision(4)<<1000*runtime[2]<<"(ms)"<<std::endl;
		std::cout<<"\t\tkernel run time: "<<std::fixed<<std::setprecision(4)<<1000*runtime[3]<<"(ms)"<<std::endl;
	}

	for(int i=0; i<4; i++)
		runtime[i]*= 1000;
	print_time(4, runtime);

	return 0;

}
