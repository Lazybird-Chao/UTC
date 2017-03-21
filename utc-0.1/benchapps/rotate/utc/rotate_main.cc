/*
 * rotate_main.cc
 *
 *  Created on: Mar 15, 2017
 *      Author: chao
 *
 *
 *
 usage:
 * 		compile with the Makefile
 * 		run as: ./a.out -v -i in.ppm -o out.ppm -a 20
 * 			-v: print time info
 * 			-t: number of threads (not useful here)
 * 			-p: number of process (not useful here)
 * 			-m: gpu memory type (0:pageable, 1:pinned, 2:unified)
 * 			-i: input image file path
 * 			-o: output image file path
 * 			-a: the angle to be rotated
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cerrno>

#include "../../common/helper_getopt.h"
#include "../../common/helper_err.h"
#include "Utc.h"
#include "UtcGpu.h"

#include "image.h"
#include "task.h"
#include "./sgpu/rotate_task_sgpu.h"


using namespace iUtc;

int main(int argc, char** argv){
	bool printTime = false;
	int nthreads=1;
	int nprocess=1;

	MemType memtype = MemType::pageable;
	int mtype = 0;
	char *infile_path=NULL;
	char *outfile_path=NULL;
	int angle=0;

	/* initialize UTC context */
	UtcContext &ctx = UtcContext::getContext(argc, argv);

	/* Parse command line options */
	int     opt;
	extern char   *optarg;
	extern int     optind;
	while ( (opt=getopt(argc,argv,"a:i:o:vt:p:m:"))!= EOF) {
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
	double runtime[4];
	Task<RotateSGPU> rotate(ProcList(0), TaskType::gpu_task);
	rotate.init(&srcImg, &dstImg, angle);
	rotate.run(runtime, MemType::pageable);
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
		std::cout<<"\tInput image size: "<<srcImg.getWidth()<<" X "<<srcImg.getHeight()<<std::endl;
		std::cout<<"\tOutput image size: "<<dstImg.getWidth()<<" X "<<dstImg.getHeight()<<std::endl;
		std::cout<<"\tTime info: "<<std::endl;
		std::cout<<"\t\tmemcpy in time: "<<std::fixed<<std::setprecision(4)<<1000*runtime[1]<<"(ms)"<<std::endl;
		std::cout<<"\t\tmemcpy out time: "<<std::fixed<<std::setprecision(4)<<1000*runtime[2]<<"(ms)"<<std::endl;
		std::cout<<"\t\tkernel run time: "<<std::fixed<<std::setprecision(4)<<1000*runtime[3]<<"(ms)"<<std::endl;
	}

	return 0;

}
