/*
 * task.cc
 *
 *  Created on: Mar 17, 2017
 *      Author: chao
 */

#include "task.h"
#include <iostream>
#include <fstream>


void ImageCreate::runImpl(Image* srcImg, char* infile_path){
	srcImg->createImageFromFile(infile_path);
}

void ImageOut::runImpl(Image * img, char* outfile_path){
	std::fstream out;
	out.open(outfile_path, std::fstream::out);
	if(!out.is_open()){
		std::cerr<<"Error, cannot create output file!"<<std::endl;
		return;
	}
	if(img->getDepth() == 3){
		out<<"P6\n";
	}
	else{
		std::cerr<<"Error, unsupported image file format!"<<std::endl;
		return;
	}
	out << img->getWidth() << " " << img->getHeight() << "\n" << img->getMaxcolor() << "\n";
	for(int i = 0; i < img->getHeight(); i++) {
		for(int j = 0; j < img->getWidth(); j++) {
			Pixel p = img->getPixelAt(j, i);
			out.put(p.r);
			out.put(p.g);
			out.put(p.b);
		}
	}
	out.close();

}


