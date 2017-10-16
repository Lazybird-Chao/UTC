/*
 * yuvconvert_main.cc
 *
 *      Author: chao
 *
 * The kernel program that convert a image from RGB color domain to
 * YUV domain. The support image file format is .ppm file.
 *
 *usage:
 *		compile with the Makefile
 *		run as: ./a.out -v -i in.ppm -o out  -l 100
 *			-v: print time info
 *			-i: input image file
 *			-o: output image file path
 *			-l: iterations of the kernel
 *
 *
 */


#include <cmath>
#include <cstring>
#include <iostream>
#include <cstdlib>
#include <iomanip>
#include "omp.h"

#include "../../common/helper_getopt.h"
#include "../../common/helper_timer.h"
#include "../../common/helper_printtime.h"
#include "image.h"

typedef struct __attribute__((aligned(4))) yuv_color{
	uint8_t *y;
	uint8_t *u;
	uint8_t *v;
}	yuv_color_t;


int main(int argc, char* argv[]){
	bool printTime = false;
	char *infile_path = nullptr;
	char *outfile_path = nullptr;
	int iterations=1;
	int nthreads = 1;

	/* Parse command line options */
	int     opt;
	extern char   *optarg;
	extern int     optind;
	while ( (opt=getopt(argc,argv,"t:l:i:o:v"))!= EOF) {
		switch (opt) {
			case 'v': printTime = true;
					  break;
			case 'i': infile_path=optarg;
					  break;
			case 'o': outfile_path = optarg;
					  break;
			case 'l': iterations = atoi(optarg);
					  break;
			case 't': nthreads = atoi(optarg);
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
	 * convert rgb to yuv
	 */
	omp_set_num_threads(nthreads);
	double t1, t2;
	t1 = getTime();
	uint8_t R,G,B,Y,U,V;
	for(int i=0; i<iterations; i++){
		uint8_t *pY = dstImg.y;
		uint8_t *pU = dstImg.u;
		uint8_t *pV = dstImg.v;
		Pixel *in = srcImg.getPixelBuffer();
#pragma omp parallel for shared(pY, pU, pV, w, h) private(R, G, B, Y, U, V)
		for(int j=0; j< w*h; j++){
			R = in[j].r;
			G = in[j].g;
			B = in[j].b;
			Y = (uint8_t)round(0.256788*R+0.504129*G+0.097906*B) + 16;
			U = (uint8_t)round(-0.148223*R-0.290993*G+0.439216*B) + 128;
			V = (uint8_t)round(0.439216*R-0.367788*G-0.071427*B) + 128;
//			*pY++ = Y;
//			*pU++ = U;
//			*pV++ = V;
			pY[j] = Y;
			pU[j] = U;
			pV[j] = V;
		}
	}
	t2 = getTime();
	double runtime = t2 - t1;


	/*
	 * write to output files
	 * each component for one file
	 */
	if(outfile_path != nullptr){
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
		std::cout<<"\tRGB->YUV runtime: "<<std::fixed<<std::setprecision(4)<<runtime*1000<<"(ms)"<<std::endl;
	}

	runtime *= 1000;
	print_time(1, &runtime);

	return 0;

}



