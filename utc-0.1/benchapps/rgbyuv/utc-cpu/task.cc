/*
 * task.cc
 *
 *  Created on: Oct 9, 2017
 *      Author: Chao
 */

#include "task.h"
#include <iostream>
#include <fstream>


void ImageCreate::runImpl(Image* srcImg, char* infile_path){
	srcImg->createImageFromFile(infile_path);
}

void ImageOut::runImpl(yuv_color_t * img, int w, int h, char* outfile_path){
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
	out << w<< " " <<h<< "\n" << 255 << "\n";
	out.write((char*)img->y, w*h);
	out.close();
	out.open(uout, std::fstream::out);
	out <<"P5\n";
	out << w<< " " <<h<< "\n" << 255 << "\n";
	out.write((char*)img->u, w*h);
	out.close();
	out.open(vout, std::fstream::out);
	out <<"P5\n";
	out << w<< " " <<h<< "\n" << 255 << "\n";
	out.write((char*)img->v, w*h);
	out.close();

}


