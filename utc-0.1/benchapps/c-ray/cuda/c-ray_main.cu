/*
 * c-ray_main.cu
 *
 * The simplified GPU ray-tracing program.
 *
 * usage:
 * 		compile with the Makefile
 * 		run as: ./a.out -i scene_file -o img_file -v -w 800 -h 600 -r 1
 * 			-v: print time info
 * 			-i: input scene file path
 * 			-o: output image file path
 * 			-w: output image x resolution
 * 			-h: output image y resolution
 * 			-r: the number of rays per pixel
 *
 */

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cerrno>
#include <cuda_runtime.h>

#include "../../common/helper_getopt.h"
#include "../../common/helper_timer.h"
#include "../../common/helper_err.h"

#include "c-ray_kernel.h"

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

int main(int argc, char**argv){
	FILE *infile = nullptr;
		FILE *outfile = nullptr;
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

		if((infile = fopen((const char*)infile_path, "rb"))==nullptr){
			std::cerr<<"Error, cannot open scene file."<<std::endl;
		}
		if((outfile = fopen((const char*)outfile_path, "wb"))==nullptr){
			std::cerr<<"Error, cannot open output file."<<std::endl;
		}

		if(!(pixels = (uint32_t*)malloc(xres * yres * sizeof *pixels))) {
			perror("pixel buffer allocation failed");
			return EXIT_FAILURE;
		}
		/*
		 * read the input scene file
		 */
		if(infile ==nullptr){
			std::cerr<<"Need a input scene file."<<std::endl;
			exit(1);
		}
		load_scene(infile);

		/* initialize the random number tables for the jitter */
		for(int i=0; i<NRAN; i++) urand[i].x = (double)rand() / RAND_MAX - 0.5;
		for(int i=0; i<NRAN; i++) urand[i].y = (double)rand() / RAND_MAX - 0.5;
		for(int i=0; i<NRAN; i++) irand[i] = (int)(NRAN * ((double)rand() / RAND_MAX));

		/*
		 * define and create necessary data structures on GPU
		 */
		unsigned int *pixels_d;
		sphere_array_t	obj_array_d;
		vec3_t	lights_d;
		vec2_t	*urand_d;
		int		*irand_d;

		checkCudaErr(cudaMalloc(&pixels_d, sizeof(unsigned int)*xres*yres));
		checkCudaErr(cudaMalloc(&(obj_array_d.pos), sizeof(vec3_t)*obj_count));
		checkCudaErr(cudaMalloc(&(obj_array_d.mat), sizeof(material_t)*obj_count));
		checkCudaErr(cudaMalloc(&(obj_array_d.rad), sizeof(FTYPE)*obj_count));
		checkCudaErr(cudaMalloc(&lights_d, sizeof(vec3_t)*MAX_LIGHTS));
		checkCudaErr(cudaMalloc(&urand_d, sizeof(vec2_t)*NRAN));
		checkCudaErr(cudaMalloc(&irand_d, sizeof(int)*NRAN));


		/*
		 * copy data in
		 */




		/*
		 * call kernel
		 */



		/*
		 * copy data out
		 */




		cudaFree(pixels_d);
		cudaFree(obj_array_d.pos);
		cudaFree(obj_array_d.mat);
		cudaFree(obj_array_d.rad);
		cudaFree(lisghts_d);
		cudaFree(urand_d);
		cudaFree(irand_d);

		/*
		 * output the image
		 */
		if(outfile != nullptr){
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

		if(printTime){
			std::cout<<"Scene info:"<<std::endl;
			std::cout<<"\tNumber of objects: "<<obj_count<<std::endl;
			std::cout<<"\tNumber of lights: "<<lnum<<std::endl;
			std::cout<<"\tTracing depth: "<<MAX_RAY_DEPTH<<std::endl;
			std::cout<<"Output image: "<<xres<<" X "<<yres<<std::endl;
			std::cout<<"Runtime: "<<std::fixed<<std::setprecision(4)<<runtime<<"(s)"<<std::endl;
		}

		return 0;
}



