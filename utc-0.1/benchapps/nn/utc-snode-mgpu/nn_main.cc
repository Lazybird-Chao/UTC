/*
 * nn_main.cc
 *
 *  Created on: Apr 12, 2017
 *      Author: Chao
 */

#include <iostream>
#include <iomanip>

#include "../../common/helper_getopt.h"
#include "../../common/helper_err.h"
#include "../../common/helper_printtime.h"
#include "Utc.h"
#include "UtcGpu.h"
using namespace iUtc;

#include "task.h"
#include "mgpu/nn_task_mgpu.h"

#define FTYPE float

int main(int argc, char **argv){
	bool printTime = false;
	int     isBinaryFile = 0;
	char   *filename = NULL;
	char *outfile = NULL;
	int numNN=1;

	int nthreads=1;
	int nprocess=1;

	MemType memtype = MemType::pageable;
	int mtype = 0;
	/* initialize UTC context */
	UtcContext &ctx = UtcContext::getContext(argc, argv);

	/* Parse command line options */
	int     opt;
	extern char   *optarg;
	extern int     optind;
	while ( (opt=getopt(argc,argv,"o:i:n:bvt:p:m:"))!= EOF) {
		switch (opt) {
			case 'v': printTime = true;
					  break;
			case 't': nthreads=atoi(optarg);
				  break;
			case 'p': nprocess = atoi(optarg);
				  break;
			case 'm': mtype = atoi(optarg);
				  break;
			case 'i': filename=optarg;
					  break;
			case 'b': isBinaryFile = 1;
					  break;
			case 'n': numNN = atoi(optarg);
					  break;
			case 'o': outfile = optarg;
					  break;
			default:
					  break;
		}
	}
	if(filename == NULL){
		std::cerr<<"Need input file path !!!"<<std::endl;
		return 1;
	}

	int procs = ctx.numProcs();
	int myproc = ctx.getProcRank();
	if(nprocess != procs){
		std::cerr<<"process number not match with arguments '-p' !!!\n";
		return 1;
	}
	/*
	if(nthreads != 1){
		std::cerr<<"require one thread !!!\n";
		return 1;
	}*/

	if(mtype==0)
		memtype = MemType::pageable;
	else if(mtype==1)
		memtype = MemType::pinned;
	else if(mtype ==2)
		memtype = MemType::unified;
	else
		std::cerr<<"wrong memory type for -m !!!"<<std::endl;

	/*
	 * init objests
	 */
	FTYPE *objects;
	int numObjs;
	int numCoords;
	Task<FileRead<FTYPE>> dataInit(ProcList(0));
	dataInit.run(isBinaryFile, filename, &objects, &numObjs, &numCoords);
	dataInit.wait();


	/*
	 * find k nearest neighbors
	 */
	FTYPE *objsNN = new FTYPE[numNN * numCoords];
	double runtime_m[8][6];
	Task<nnMGPU<FTYPE>> nnCompute(ProcList(nthreads, 0), TaskType::gpu_task);
	nnCompute.init(objects, objsNN, numObjs, numCoords, numNN);
	nnCompute.run(runtime_m, memtype);
	nnCompute.wait();
	double runtime[6]={0,0,0,0,0,0};
	for(int i=0; i<nthreads; i++)
		for(int j=0; j<6; j++)
			runtime[j]+= runtime_m[i][j];
	for(int j=0; j<6; j++)
		runtime[j] /= nthreads;

	/*
	 * output
	 */
	if(outfile != NULL){
		Task<Output<FTYPE>> fileout(ProcList(0));
		fileout.run(outfile, objsNN, numNN, numCoords);
		fileout.wait();
	}

	if(objects)
		free(objects);
	if(objsNN)
		delete objsNN;

	std::cout<<"Test complete !!!"<<std::endl;
	if(printTime){
		std::cout<<"Data info:"<<std::endl;
		std::cout<<"\tnumObjs = "<<numObjs<<std::endl;
		std::cout<<"\tnumCoords = "<<numCoords<<std::endl;
		std::cout<<"\tumNN = "<<numNN<<std::endl;
		std::cout<<"Time info:"<<std::endl;
		std::cout<<"kernel1 time: "<<std::fixed<<std::setprecision(4)<<runtime[1]*1000<<std::endl;
		std::cout<<"kernel2 time: "<<std::fixed<<std::setprecision(4)<<runtime[2]*1000<<std::endl;
		std::cout<<"host compute time: "<<std::fixed<<std::setprecision(4)<<runtime[5]*1000<<std::endl;
		std::cout<<"copyin time "<<std::fixed<<std::setprecision(4)<<runtime[3]*1000<<std::endl;
		std::cout<<"copyout time "<<std::fixed<<std::setprecision(4)<<runtime[4]*1000<<std::endl;
	}

	for(int i=0; i<6; i++)
		runtime[i] *= 1000;

	print_time(6, runtime);

	return 0;

}









