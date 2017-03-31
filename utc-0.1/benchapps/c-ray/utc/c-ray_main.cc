/*
 * c-ray_main.cc
 *
 *  Created on: Mar 24, 2017
 *      Author: chao
 */


#include <iostream>
#include <iomanip>
#include <cmath>
#include <cerrno>
#include <stdint.h>

#include "../../common/helper_getopt.h"
#include "../../common/helper_err.h"
#include "../../common/helper_printtime.h"
#include "Utc.h"
#include "UtcGpu.h"
using namespace iUtc;

#include "sgpu/c-ray_task_sgpu.h"
#include "task.h"
#include "typeconfig.h"

FTYPE aspect = 1.333333;

int main(int argc, char** argv){
	bool printTime = false;
	char* infile_path = NULL;
	char* outfile_path = NULL;
	int xres=800;
	int yres=600;
	int rays_per_pixel=1;

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
	while ( (opt=getopt(argc,argv,"w:h:r:i:o:vt:p:m:"))!= EOF) {
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
			case 'w': xres = atoi(optarg);
					  break;
			case 'h': yres = atoi(optarg);
					  break;
			case 'r': rays_per_pixel = atoi(optarg);
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

	/*
	 * read input scene file
	 */
	global_vars g_vars;
	g_vars.xres = xres;
	g_vars.yres = yres;
	g_vars.rays_per_pixel = rays_per_pixel;
	g_vars.aspect = aspect;
	sphere_array_t obj_array;
	vec3_t lights[MAX_LIGHTS];
	Task<SceneInit> sceneConfig(ProcList(0));
	sceneConfig.run(infile_path, &obj_array, &g_vars, lights);
	sceneConfig.wait();
	uint32_t *pixels = (uint32_t*)malloc(xres * yres * sizeof *pixels);

	/*
	 * tracing
	 */
	double runtime[4];
	Task<craySGPU> cray(ProcList(0), TaskType::gpu_task);
	cray.init(g_vars, obj_array, pixels, lights);
	cray.run(runtime, memtype);
	cray.wait();

	/*
	 * output image
	 */
	if(outfile_path){
		Task<Output> fileOut(ProcList(0));
		fileOut.run(outfile_path, pixels, xres, yres);
		fileOut.wait();
	}

	if(obj_array.mat)
		free(obj_array.mat);
	if(obj_array.pos)
		free(obj_array.pos);
	if(obj_array.rad)
		free(obj_array.rad);
	if(pixels)
		free(pixels);

	std::cout<<"Test complete !!!"<<std::endl;
	if(printTime){
		std::cout<<"Scene info:"<<std::endl;
		std::cout<<"\tNumber of objects: "<<g_vars.obj_count<<std::endl;
		std::cout<<"\tNumber of lights: "<<g_vars.lnum<<std::endl;
		std::cout<<"\tTracing depth: "<<MAX_RAY_DEPTH<<std::endl;
		std::cout<<"\tRays per pixel: "<<rays_per_pixel<<std::endl;
		std::cout<<"Output image: "<<xres<<" X "<<yres<<std::endl;
		std::cout<<"Total time: "<<std::fixed<<std::setprecision(4)<<runtime[0]<<std::endl;
		std::cout<<"Kernel Runtime: "<<std::fixed<<std::setprecision(4)<<runtime[1]<<"(s)"<<std::endl;
		std::cout<<"copy in time: "<<std::fixed<<std::setprecision(4)<<runtime[2]<<"(s)"<<std::endl;
		std::cout<<"copy out time: "<<std::fixed<<std::setprecision(4)<<runtime[3]<<"(s)"<<std::endl;
	}

	for(int i=0; i<4; i++)
		runtime[i] *= 1000;
	print_time(4, runtime);


	return 0;
}
