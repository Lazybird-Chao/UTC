/*
 * c-ray_main.cc
 *
 *  Created on: Oct 10, 2017
 *      Author: chaoliu
 */
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cerrno>
#include <stdint.h>

#include "../../common/helper_getopt.h"
#include "../../common/helper_printtime.h"
#include "Utc.h"
using namespace iUtc;

#include "task.h"
#include "typeconfig.h"

#define MAX_THREADS 64

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

	/* initialize UTC context */
	UtcContext &ctx = UtcContext::getContext(argc, argv);
	std::cout<<"UTC context initialized !"<<std::endl;

	/* Parse command line options */
	int     opt;
	extern char   *optarg;
	extern int     optind;
	while ( (opt=getopt(argc,argv,"w:h:r:i:o:vt:p:"))!= EOF) {
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
	if(nprocess != procs || nprocess > 1){
		std::cerr<<"process number not match with arguments '-p' !!!\n";
		return 1;
	}
	/*if(nthreads != 1){
		std::cerr<<"require one thread !!!\n";
		return 1;
	}*/

	/*
	 * read input scene file
	 */
	global_vars g_vars;
	g_vars.xres = xres;
	g_vars.yres = yres;
	g_vars.rays_per_pixel = rays_per_pixel;
	g_vars.aspect = aspect;
	sphere2_t *obj_array;
	vec3_t lights[MAX_LIGHTS];
	Task<SceneInit> sceneConfig(ProcList(0));
	sceneConfig.run(infile_path, &obj_array, &g_vars, lights);
	sceneConfig.wait();
	uint32_t *pixels = (uint32_t*)malloc(xres * yres * sizeof *pixels);

	/*
	 * tracing
	 */
	double runtime_m[MAX_THREADS][1];
	Task<crayWorker> cray(ProcList(nthreads, 0));
	cray.init(g_vars, obj_array, pixels, lights);
	cray.run(runtime_m);
	cray.wait();

	/*
	 * output image
	 */
	if(outfile_path){
		Task<Output> fileOut(ProcList(0));
		fileOut.run(outfile_path, pixels, xres, yres);
		fileOut.wait();
	}

	if(obj_array)
		free(obj_array);
	if(pixels)
		free(pixels);

	double runtime = 0.0;
	for(int i=0; i<nthreads; i++)
		runtime += runtime_m[i][0];
	runtime /= nthreads;
	std::cout<<"Test complete !!!"<<std::endl;
	if(printTime){
		std::cout<<"Scene info:"<<std::endl;
		std::cout<<"\tNumber of objects: "<<g_vars.obj_count<<std::endl;
		std::cout<<"\tNumber of lights: "<<g_vars.lnum<<std::endl;
		std::cout<<"\tTracing depth: "<<MAX_RAY_DEPTH<<std::endl;
		std::cout<<"\tRays per pixel: "<<rays_per_pixel<<std::endl;
		std::cout<<"Output image: "<<xres<<" X "<<yres<<std::endl;
		std::cout<<"Total time: "<<std::fixed<<std::setprecision(4)<<runtime<<std::endl;
	}

	runtime *= 1000;
	print_time(1, &runtime);

	return 0;
}

