/*
 * rotate_main.cu
 *
 *      Author: chao
 *
 * Basic gpu implementation of image rotate program.
 *
 * usage:
 * 		compile with the Makefile
 * 		run as: ./a.out -v -i in.ppm -o out.ppm -a 20
 * 			-v: print time info
 * 			-i: input image file path
 * 			-o: output image file path
 * 			-a: the angle to will be rotated
 */


#include <iostream>
#include <iomanip>
#include <fstream>
#include <cerrno>
#include <cuda_runtime.h>

#include "../../common/helper_getopt.h"
#include "../../common/helper_timer.h"
#include "../../common/helper_err.h"
#include "image.h"
#include "rotate_kernel.h"


int main(int argc, char* argv[]){
	bool printTime = false;
	char *infile_path=NULL;
	char *outfile_path=NULL;
	int angle=0;

	/* Parse command line options */
	int     opt;
	extern char   *optarg;
	extern int     optind;
	while ( (opt=getopt(argc,argv,"a:i:o:v"))!= EOF) {
		switch (opt) {
			case 'v': printTime = true;
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

	if(infile_path == NULL){
		std::cerr<<"Error, need the input image file."<<std::endl;
		return 1;
	}


	/*
	 * create image object from the file
	 */
	Image srcImg;
	srcImg.createImageFromFile(infile_path);
	Image dstImg;

	if(angle == 0 || angle == 360){
		std::cout<<"No need to rotate, same as original image."<<std::endl;
		return 0;
	}

	/*
	 * compute the out image's size
	 */
	float2 ul, ur, ll, lr;
	float xc = (float)srcImg.getWidth() / 2.0;
	float yc = (float)srcImg.getHeight() / 2.0;
	ul.x = -xc;
	ul.y = yc;
	ur.x = xc;
	ur.y = yc;
	ll.x = -xc;
	ll.y = -yc;
	lr.x = xc;
	lr.y = -yc;
	float2 outCorner[4];
	rotatePoint(ul, outCorner[0], angle);
	rotatePoint(ur, outCorner[1], angle);
	rotatePoint(ll, outCorner[2], angle);
	rotatePoint(lr, outCorner[3], angle);
	//compute the out image's size
	float maxW = outCorner[0].x;
	float minW = outCorner[0].x;
	float maxH = outCorner[0].y;
	float minH = outCorner[0].y;
	for(int i=1; i<4; i++){
		if(outCorner[i].x > maxW)
			maxW = outCorner[i].x;
		if(outCorner[i].x< minW)
			minW = outCorner[i].x;
		if(outCorner[i].y > maxH)
			maxH = outCorner[i].y;
		if(outCorner[i].y< minH)
			minH = outCorner[i].y;
	}
	int outH = (int)maxH-minH;
	int outW = (int)maxW-minW;
	dstImg.createImageFromTemplate(outW, outH, srcImg.getDepth());

	checkCudaErr(cudaSetDevice(0));

	double t1, t2;
	/*
	 * crate gpu memory
	 */
	Pixel *srcImgData_d;
	Pixel *dstImgData_d;
	checkCudaErr(cudaMalloc(
			&srcImgData_d,
			sizeof(Pixel)*srcImg.getWidth()*srcImg.getHeight()));
	checkCudaErr(cudaMalloc(
			&dstImgData_d,
			sizeof(Pixel)*dstImg.getWidth()*dstImg.getHeight()));

	/*
	 * copy data in
	 */
	t1 = getTime();
	checkCudaErr(cudaMemcpy(
			srcImgData_d, srcImg.getPixelBuffer(),
			sizeof(Pixel)*srcImg.getWidth()*srcImg.getHeight(),
			cudaMemcpyHostToDevice));
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
	dim3 grid((outW+blocksize*batchx-1)/(blocksize*batchx),
				(outH+blocksize*batchy-1)/(blocksize*batchy),
				1);
	rotate_kernel<<<grid, block>>>(srcImgData_d,
									srcImg.getWidth(),
									srcImg.getHeight(),
									dstImgData_d,
									outW,
									outH,
									angle,
									batchx,
									batchy);
	checkCudaErr(cudaGetLastError());
	checkCudaErr(cudaDeviceSynchronize());
	t2 = getTime();
	double kernelTime = t2-t1;

	/*
	 * copy data out
	 */
	t1 = getTime();
	checkCudaErr(cudaMemcpy(
			dstImg.getPixelBuffer(), dstImgData_d,
			sizeof(Pixel)*dstImg.getWidth()*dstImg.getHeight(),
			cudaMemcpyDeviceToHost));
	t2 = getTime();
	double copyoutTime = t2-t1;

	cudaFree(srcImgData_d);
	cudaFree(dstImgData_d);
	/*
	 * out put the image
	 */
	if(outfile_path != NULL){
		std::fstream out;
		out.open(outfile_path, std::fstream::out);
		if(!out.is_open()){
			std::cerr<<"Error, cannot create output file!"<<std::endl;
			return 1;
		}
		if(dstImg.getDepth() == 3){
			out<<"P6\n";
		}
		else{
			std::cerr<<"Error, unsupported image file format!"<<std::endl;
			return 1;
		}
		out << dstImg.getWidth() << " " << dstImg.getHeight() << "\n" << dstImg.getMaxcolor() << "\n";
		for(int i = 0; i < dstImg.getHeight(); i++) {
			for(int j = 0; j < dstImg.getWidth(); j++) {
				Pixel p = dstImg.getPixelAt(j, i);
				out.put(p.r);
				out.put(p.g);
				out.put(p.b);
			}
		}
		out.close();
	}

	srcImg.clean();
	dstImg.clean();

	std::cout<<"Test complete !!!"<<std::endl;
	if(printTime){
		std::cout<<"\tInput image size: "<<srcImg.getWidth()<<" X "<<srcImg.getHeight()<<std::endl;
		std::cout<<"\tOutput image size: "<<dstImg.getWidth()<<" X "<<dstImg.getHeight()<<std::endl;
		std::cout<<"\tTime info: "<<std::endl;
		std::cout<<"\t\tmemcpy in time: "<<std::fixed<<std::setprecision(4)<<copyinTime<<"(s)"<<std::endl;
		std::cout<<"\t\tmemcpy out time: "<<std::fixed<<std::setprecision(4)<<copyoutTime<<"(s)"<<std::endl;
		std::cout<<"\t\tkernel run time: "<<std::fixed<<std::setprecision(4)<<kernelTime<<"(s)"<<std::endl;
	}

	return 0;
}





