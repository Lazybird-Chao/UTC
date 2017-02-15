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

#include "../../common/helper_getopt.h"
#include "../../common/helper_timer.h"
#include "image.h"
#include "rotation.h"


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
	}

	Image srcImg;
	// create img object from the file
	srcImg.createImageFromFile(infile_path);
	Image dstImg;

	// do image rotation
	double t1, t2;
	t1 = getTime();
	rotation(srcImg, dstImg, angle);
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

	if(printTime){
		std::cout<<"Input image size: "<<srcImg.getWidth()<<" X "<<srcImg.getHeight()<<std::endl;
		std::cout<<"Rotation runtime: "<<std::fixed<<std::setprecision(4)<<runtime<<"(s)"<<std::endl;
	}
	return 0;

}





