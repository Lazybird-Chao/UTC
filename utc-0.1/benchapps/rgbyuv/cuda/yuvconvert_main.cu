/*
 * yuvconvert_main.cu
 *
 *      Author: chao
 *
 * Basic gpu implementation of converting image RGB clolor to YUV color.
 *
 * usage:
 * 		compile with the Makefile
 *		run as: ./a.out -v -i in.ppm -o out -a 20  -l 100
 *			-v: print time info
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
#include "../../common/helper_timer.h"
#include "../../common/helper_err.h"
#include "image.h"
#include "yuvconvert_kernel.h"




int main(int argc, char* argv[]){
	bool printTime = false;
	char *infile_path = NULL;
	char *outfile_path = NULL;
	int iterations=1;

	/* Parse command line options */
	int     opt;
	extern char   *optarg;
	extern int     optind;
	while ( (opt=getopt(argc,argv,"l:i:o:v"))!= EOF) {
		switch (opt) {
			case 'v': printTime = true;
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

	if(infile_path == NULL){
		std::cerr<<"Error, need the input image file."<<std::endl;
		return 1;
	}

	/*
	 * create img object from the input file
	 */
	Image srcImg;
	srcImg.createImageFromFile(infile_path);
	int w = srcImg.getWidth();
	int h = srcImg.getHeight();
	yuv_color_t dstImg;
	dstImg.y = (uint8_t*)malloc(sizeof(uint8_t)*w*h);
	dstImg.u = (uint8_t*)malloc(sizeof(uint8_t)*w*h);
	dstImg.v = (uint8_t*)malloc(sizeof(uint8_t)*w*h);


	/*
	 * create gpu memory
	 */
	Pixel *srcImgData_d;
	yuv_color_t dstImgData_d;
	checkCudaErr(cudaMalloc(
			&srcImgData_d,
			sizeof(Pixel)*w*h));
	checkCudaErr(cudaMalloc(
			&dstImgData_d.y, sizeof(uint8_t)*w*h));
	checkCudaErr(cudaMalloc(
			&dstImgData_d.u, sizeof(uint8_t)*w*h));
	checkCudaErr(cudaMalloc(
			&dstImgData_d.v, sizeof(uint8_t)*w*h));

	/*
	 * copy data in
	 */
	double t1, t2;
	t1 = getTime();
	checkCudaErr(cudaMemcpy(
			srcImgData_d, srcImg.getPixelBuffer(),
			sizeof(Pixel)*w*h, cudaMemcpyHostToDevice));
	t2 = getTime();
	double copyinTime = t2 -t1;

	/*
	 * invoke kernel
	 */
	t1 = getTime();
	int blocksize = 16;
	int batchx = 1;
	int batchy = 1;
	dim3 block(blocksize, blocksize, 1);
	dim3 grid((w+blocksize*batchx-1)/(blocksize*batchx),
				(h+blocksize*batchy-1)/(blocksize*batchy),
				1);
	for(int i=0; i<iterations; i++){
		convert<<<grid, block>>>(srcImgData_d,
				dstImgData_d,
				h, w, batchx, batchy);
		checkCudaErr(cudaGetLastError());
		checkCudaErr(cudaDeviceSynchronize());
	}
	t2 = getTime();
	double kernelTime = t2-t1;

	/*
	 * copy data out
	 */
	t1 = getTime();
	checkCudaErr(cudaMemcpy(
			dstImg.y, dstImgData_d.y, sizeof(uint8_t)*w*h, cudaMemcpyDeviceToHost));
	checkCudaErr(cudaMemcpy(
			dstImg.u, dstImgData_d.u, sizeof(uint8_t)*w*h, cudaMemcpyDeviceToHost));
	checkCudaErr(cudaMemcpy(
			dstImg.v, dstImgData_d.v, sizeof(uint8_t)*w*h, cudaMemcpyDeviceToHost));
	t2 = getTime();
	double copyoutTime = t2-t1;

	cudaFree(srcImgData_d);
	cudaFree(dstImgData_d.y);
	cudaFree(dstImgData_d.u);
	cudaFree(dstImgData_d.v);

	/*
	 * output to file
	 */
	if(outfile_path != NULL){
		char yout[256];
		char uout[256];
		char vout[256];
		strcpy(yout, outfile_path);
		strcpy(uout, outfile_path);
		strcpy(vout, outfile_path);
		strcat(yout, "_y.ppm");
		strcat(uout, "_u.ppm");
		strcat(vout, "_v.ppm");

		std::fstream out;
		out.open(yout, std::fstream::out);
		out <<"P5\n";
		out << w<< " " <<h<< "\n" << srcImg.getMaxcolor() << "\n";
		out.write((char*)dstImg.y, w*h);
		out.close();
		out.open(uout, std::fstream::out);
		out <<"P5\n";
		out << w<< " " <<h<< "\n" << srcImg.getMaxcolor() << "\n";
		out.write((char*)dstImg.u, w*h);
		out.close();
		out.open(vout, std::fstream::out);
		out <<"P5\n";
		out << w<< " " <<h<< "\n" << srcImg.getMaxcolor() << "\n";
		out.write((char*)dstImg.v, w*h);
		out.close();
	}

	srcImg.clean();
	free(dstImg.y);
	free(dstImg.u);
	free(dstImg.v);
	std::cout<<"Test complete !!!"<<std::endl;
	if(printTime){
		std::cout<<"\tInput image size: "<<srcImg.getWidth()<<" X "<<srcImg.getHeight()<<std::endl;
		std::cout<<"\tTime info: "<<std::endl;
		std::cout<<"\t\tmemcpy in time: "<<std::fixed<<std::setprecision(4)<<copyinTime<<"(s)"<<std::endl;
		std::cout<<"\t\tmemcpy out time: "<<std::fixed<<std::setprecision(4)<<copyoutTime<<"(s)"<<std::endl;
		std::cout<<"\t\tkernel run time: "<<std::fixed<<std::setprecision(4)<<kernelTime<<"(s)"<<std::endl;
	}
	return 0;

}


