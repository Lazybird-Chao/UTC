/*
 * rotate_main.cc
 *
 *      Author: chao
 *
 * Image rotation program.
 * Take a .ppm image file as input, and rotate angle degree based on
 * center of the original image.
 *
 * usage:
 * 		compile with the Makefile
 * 		run as: ./a.out -v -i in.ppm -o out.ppm -a 20
 * 			-v: print time info
 * 			-i: input image file path
 * 			-o: output image file path
 * 			-a: the angle to will be rotated
 *
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cerrno>
#include "omp.h"

#include "../../common/helper_getopt.h"
#include "../../common/helper_timer.h"
#include "../../common/helper_printtime.h"
#include "image.h"
#include "rotation.h"


int main(int argc, char* argv[]){
	bool printTime = false;
	char *infile_path=NULL;
	char *outfile_path=NULL;
	int angle=0;
	int nthreads = 1;

	/* Parse command line options */
	int     opt;
	extern char   *optarg;
	extern int     optind;
	while ( (opt=getopt(argc,argv,"a:i:o:vt:"))!= EOF) {
		switch (opt) {
			case 'v': printTime = true;
					  break;
			case 'i': infile_path=optarg;
					  break;
			case 'o': outfile_path = optarg;
					  break;
			case 'a': angle = atoi(optarg);
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
	 * create img object from the file
	 */
	Image srcImg;
	srcImg.createImageFromFile(infile_path);
	Image dstImg;

	/*
	 * do image rotation
	 */
	//omp_set_dynamic(0);
	//omp_set_num_threads(nthreads);
	double t1, t2;
	t1 = getTime();
	rotation(srcImg, dstImg, angle, nthreads);
	t2 = getTime();
	double runtime = t2-t1;

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
		std::cout<<"\tRotation runtime: "<<std::fixed<<std::setprecision(4)<<1000*runtime<<"(ms)"<<std::endl;
	}

	runtime *=1000;
	print_time(1, &runtime);

	return 0;

}





