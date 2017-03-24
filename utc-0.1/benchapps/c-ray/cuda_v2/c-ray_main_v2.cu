/*
 * c-ray_main.cu
 *
 * The simplified GPU ray-tracing program.
 * For this version, we use some __device__ or __const__ global
 * arrays and variables instead of always pass them as arguments for device functions.
 *
 * usage:
 * 		compile with the Makefile
 * 		run as: ./a.out -i scene_file -o img_file -v -w 800 -h 600 -r 1
 * 			-v: print time info
 * 			-i: input scene file path
 * 			-o: output image file path, with .ppm suffix
 * 			-w: output image x resolution
 * 			-h: output image y resolution
 * 			-r: the number of rays per pixel
 *
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cerrno>
#include <stdint.h>
#include <cuda_runtime.h>

#include "../../common/helper_getopt.h"
#include "../../common/helper_timer.h"
#include "../../common/helper_err.h"

#include "c-ray_kernel_v2.h"

/*
 * global variables
 */
vec2_t urand[NRAN];
int irand[NRAN];
int xres = 800;
int yres = 600;
int rays_per_pixel = 1;
FTYPE aspect = 1.333333;
sphere_t *obj_list;
vec3_t lights[MAX_LIGHTS];
int lnum = 0;
camera_t cam;
sphere_array_t obj_array; // used to replace the obj_list
int obj_count;

__device__  global_vars g_vars_d;
__device__  vec3_t lights_d[MAX_LIGHTS];
__device__  vec2_t urand_d[NRAN];
__device__  int irand_d[NRAN];

void load_scene(FILE *fp);

int main(int argc, char**argv){
	FILE *infile = NULL;
		FILE *outfile = NULL;
		uint32_t *pixels;
		bool printTime = false;
		char* infile_path;
		char* outfile_path;

		/* Parse command line options */
		int     opt;
		extern char   *optarg;
		extern int     optind;
		while ( (opt=getopt(argc,argv,"w:h:r:i:o:v"))!= EOF) {
			switch (opt) {
				case 'v': printTime = true;
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

		if((infile = fopen((const char*)infile_path, "rb"))==NULL){
			std::cerr<<"Error, cannot open scene file."<<std::endl;
		}
		if((outfile = fopen((const char*)outfile_path, "wb"))==NULL){
			std::cerr<<"Error, cannot open output file."<<std::endl;
		}

		if(!(pixels = (uint32_t*)malloc(xres * yres * sizeof *pixels))) {
			perror("pixel buffer allocation failed");
			return EXIT_FAILURE;
		}
		/*
		 * read the input scene file
		 */
		if(infile ==NULL){
			std::cerr<<"Need a input scene file."<<std::endl;
			exit(1);
		}
		load_scene(infile);

		/* initialize the random number tables for the jitter */
		for(int i=0; i<NRAN; i++) urand[i].x = (double)rand() / RAND_MAX - 0.5;
		for(int i=0; i<NRAN; i++) urand[i].y = (double)rand() / RAND_MAX - 0.5;
		for(int i=0; i<NRAN; i++) irand[i] = (int)(NRAN * ((double)rand() / RAND_MAX));
		struct global_vars g_vars;
		g_vars.aspect = aspect;
		g_vars.cam = cam;
		g_vars.lnum = lnum;
		g_vars.obj_count = obj_count;
		g_vars.xres = xres;
		g_vars.yres = yres;
		g_vars.rays_per_pixel = rays_per_pixel;

		cudaSetDevice(0);

		/*
		 * define and create necessary data structures on GPU
		 */
		unsigned int *pixels_d;
		sphere_array_t	obj_array_d;
		checkCudaErr(cudaMalloc(&pixels_d, sizeof(unsigned int)*xres*yres));
		checkCudaErr(cudaMalloc(&(obj_array_d.pos), sizeof(vec3_t)*obj_count));
		checkCudaErr(cudaMalloc(&(obj_array_d.mat), sizeof(material_t)*obj_count));
		checkCudaErr(cudaMalloc(&(obj_array_d.rad), sizeof(FTYPE)*obj_count));


		/*
		 * copy data in
		 */
		double copyinTime = 0;
		double t1, t2;
		t1 = getTime();
		checkCudaErr(
				cudaMemcpy(obj_array_d.pos, obj_array.pos, sizeof(vec3_t)*obj_count,
						cudaMemcpyHostToDevice));
		checkCudaErr(
				cudaMemcpy(obj_array_d.mat, obj_array.mat, sizeof(material_t)*obj_count,
						cudaMemcpyHostToDevice));
		checkCudaErr(
				cudaMemcpy(obj_array_d.rad, obj_array.rad, sizeof(FTYPE)*obj_count,
						cudaMemcpyHostToDevice));
		checkCudaErr(
				cudaMemcpyToSymbol(lights_d, lights, sizeof(vec3_t)*MAX_LIGHTS, 0,
						cudaMemcpyHostToDevice));
		checkCudaErr(
				cudaMemcpyToSymbol(urand_d, urand, sizeof(vec2_t)*NRAN, 0,
						cudaMemcpyHostToDevice));
		checkCudaErr(
				cudaMemcpyToSymbol(irand_d, irand, sizeof(int)*NRAN, 0,
						cudaMemcpyHostToDevice));
		checkCudaErr(
				cudaMemcpyToSymbol(g_vars_d, (void*)&g_vars, sizeof(g_vars), 0,
						cudaMemcpyHostToDevice));
		t2 = getTime();
		copyinTime = t2-t1;


		/*
		 * call kernel
		 *
		 * kernel has a recursive function call and may call stack overflow,
		 * so need to set the cuda user stack frame to a larger size
		 *
		 * perthread stack size should < local memory size(64KB) and
		 * 							   < (gpu_mem_size)/(#sm_of_gpu)/(#threads_per_sm)
		 *
		 */
		size_t stacksize;
		cudaThreadGetLimit(&stacksize, cudaLimitStackSize);
		//std::cout<<stacksize<<std::endl;  //defaut is 1024
		stacksize = 1024*4;
		cudaThreadSetLimit(cudaLimitStackSize, stacksize);
		dim3 block(16, 8, 1);
		dim3 grid((xres+block.x-1)/block.x, (yres+block.y-1)/block.y,1);
		double kernelrunTime = 0;
		t1= getTime();
		render_kernel<<<grid, block>>>(
				pixels_d,
				obj_array_d);
		checkCudaErr(cudaDeviceSynchronize());
		t2 = getTime();
		kernelrunTime =t2-t1;

		/*
		 * copy data out
		 */
		double copyoutTime = 0;
		t1 = getTime();
		checkCudaErr(
				cudaMemcpy(pixels, pixels_d, sizeof(unsigned int)*xres*yres,
						cudaMemcpyDeviceToHost));
		t2 = getTime();
		copyoutTime = t2 - t1;


		cudaFree(pixels_d);
		cudaFree(obj_array_d.pos);
		cudaFree(obj_array_d.mat);
		cudaFree(obj_array_d.rad);

		cudaDeviceReset();

		/*
		 * output the image
		 */
		if(outfile != NULL){
			fprintf(outfile, "P6\n%d %d\n255\n", xres, yres);
			for(int i=0; i<xres * yres; i++) {
				fputc((pixels[i] >> RSHIFT) & 0xff, outfile);
				fputc((pixels[i] >> GSHIFT) & 0xff, outfile);
				fputc((pixels[i] >> BSHIFT) & 0xff, outfile);
			}
			fflush(outfile);
		}
		if(infile) fclose(infile);
		if(outfile) fclose(outfile);
		if(obj_array.pos){
			free(obj_array.pos);
			free(obj_array.mat);
			free(obj_array.rad);
		}
		free(pixels);

		double totaltime = kernelrunTime + copyinTime + copyoutTime;
		if(printTime){
			std::cout<<"Scene info:"<<std::endl;
			std::cout<<"\tNumber of objects: "<<obj_count<<std::endl;
			std::cout<<"\tNumber of lights: "<<lnum<<std::endl;
			std::cout<<"\tTracing depth: "<<MAX_RAY_DEPTH<<std::endl;
			std::cout<<"\tRays per pixel: "<<rays_per_pixel<<std::endl;
			std::cout<<"Output image: "<<xres<<" X "<<yres<<std::endl;
			std::cout<<"Total time: "<<std::fixed<<std::setprecision(4)<<totaltime<<std::endl;
			std::cout<<"Kernel Runtime: "<<std::fixed<<std::setprecision(4)<<kernelrunTime<<"(s)"<<std::endl;
			std::cout<<"copy in time: "<<std::fixed<<std::setprecision(4)<<copyinTime<<"(s)"<<std::endl;
			std::cout<<"copy out time: "<<std::fixed<<std::setprecision(4)<<copyoutTime<<"(s)"<<std::endl;
		}

		return 0;
}

/* Load the scene from an extremely simple scene description file */
#define DELIM	" \t\n"
void load_scene(FILE *fp) {
	char line[256], *ptr, type;

	obj_list = (sphere_t*)malloc(sizeof(struct sphere));
	obj_list->next = NULL;
	obj_count=0;

	while((ptr = fgets(line, 256, fp))) {
		int i;
		vec3_t pos, col;
		FTYPE rad, spow, refl;

		while(*ptr == ' ' || *ptr == '\t') ptr++;
		if(*ptr == '#' || *ptr == '\n') continue;

		if(!(ptr = strtok(line, DELIM))) continue;
		type = *ptr;

		for(i=0; i<3; i++) {
			if(!(ptr = strtok(0, DELIM))) break;
			*((FTYPE*)&pos.x + i) = (FTYPE)atof(ptr);
		}

		if(type == 'l') {
			lights[lnum++] = pos;
			continue;
		}

		if(!(ptr = strtok(0, DELIM))) continue;
		rad = atof(ptr);

		for(i=0; i<3; i++) {
			if(!(ptr = strtok(0, DELIM))) break;
			*((FTYPE*)&col.x + i) = (FTYPE)atof(ptr);
		}

		if(type == 'c') {
			cam.pos = pos;
			cam.targ = col;
			cam.fov = rad;
			continue;
		}

		if(!(ptr = strtok(0, DELIM))) continue;
		spow = (FTYPE)atof(ptr);

		if(!(ptr = strtok(0, DELIM))) continue;
		refl = (FTYPE)atof(ptr);

		if(type == 's') {
			obj_count++;
			struct sphere *sph = (sphere_t*)malloc(sizeof *sph);
			sph->next = obj_list->next;
			obj_list->next = sph;

			sph->pos = pos;
			sph->rad = rad;
			sph->mat.col = col;
			sph->mat.spow = spow;
			sph->mat.refl = refl;
		} else {
			fprintf(stderr, "unknown type: %c\n", type);
		}
	}

	/*
	 * change the sphere linked list to an array
	 */
	obj_array.pos = (vec3_t*)malloc(sizeof(vec3_t)*obj_count);
	obj_array.mat = (material_t*)malloc(sizeof(material_t)*obj_count);
	obj_array.rad = (FTYPE*)malloc(sizeof(FTYPE)*obj_count);
	sphere_t *p1 = obj_list->next;
	sphere_t *p2 = p1;
	int i=0;
	while(p1!=NULL){
		obj_array.pos[i] = p1->pos;
		obj_array.rad[i] = p1->rad;
		obj_array.mat[i].col = p1->mat.col;
		obj_array.mat[i].spow = p1->mat.spow;
		obj_array.mat[i].refl = p1->mat.refl;
		p2 = p1;
		p1 = p1->next;
		free(p2);
		i++;
	}
	obj_list->next = NULL;
	free(obj_list);
}











