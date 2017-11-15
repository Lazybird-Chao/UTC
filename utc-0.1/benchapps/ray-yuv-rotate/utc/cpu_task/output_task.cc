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

void OutputWorker::runImpl(int loop, iUtc::Conduit *cdtIn, double runtime[][3], int id){
	if(__localThreadId == 0){
			std::cout<<getCurrentTask()->getName()<<" begin run ..."<<std::endl;
		}
	Timer timer, timer0;
	double commtime = 0;
	double comptime = 0;
	int iter = 0;
	timer0.start();
	while(iter < loop){
		timer.start();
		int imgSize[2];
		//std::cout<<"output read size"<<std::endl;
		cdtIn->Read(imgSize, 2*sizeof(int), iter*2);
		int w = imgSize[0];
		int h = imgSize[1];
		//std::cout<<"output:"<<w<<" "<<h<<std::endl;
		Pixel *img = new Pixel[w*h];
		cdtIn->Read(img, w*h*sizeof(Pixel), iter*2+1);
		commtime += timer.stop();

		timer.start();
		std::string outfile_path = "./outfile/";
		outfile_path += std::to_string(iter*3+id);
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
		comptime += timer.stop();
		std::cout<<"iter "<<iter<<"finish..."<<std::endl;
		iter++;
	}
	runtime[__localThreadId][0] = timer0.stop();
	runtime[__localThreadId][1] = comptime;
	runtime[__localThreadId][2] = commtime;
	if(__localThreadId == 0){
		std::cout<<getCurrentTask()->getName()<<" finish run ..."<<std::endl;
	}
}


