/*
 * output_task.cc
 *
 *  Created on: Nov 14, 2017
 *      Author: chaoliu
 */

#include "output_task.h"
#include <iostream>
#include <fstream>
#include <cstring>

void OutputWorker::runImpl(int w, int h, int loop, iUtc::Conduit cdtIn, double runtime[][1]){
	Pixel *img = new Pixel[w*h];
	timer timer;
	int iter = 0;
	timer.start();
	while(iter < 0){

		std::string outfile_path = "./outfile/";
		outfile_path += std::to_string(iter);
		outfile_path += ".ppm";
		std::fstream out;
		out.open(outfile_path.c_str(), std::fstream::out);
		out<<"P6\n";
		out << w << " " << h << "\n" << 255 << "\n";
		for(int i = 0; i < h; i++) {
			for(int j = 0; j < w; j++) {
				Pixel p = img[i*w + j];
				out.put(p.r);
				out.put(p.g);
				out.put(p.b);
			}
		}
		out.close();
	}
	runtime[__localThreadId][0] = timer.stop();
}


