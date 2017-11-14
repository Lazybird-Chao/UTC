/*
 * rotate.cc
 *
 *  Created on: Nov 13, 2017
 *      Author: Chao
 */

#include "image.h"
#include "rotation.h"

#include <stdint.h>
#include <stdlib.h>

void rotate(int w, int h, uint8_t *yuv, Image **dstImg){
	Image srcImg;
	Pixel *sImgBuffer = new Pixel[w*h];

	//std::cout<<w<<" "<<h<<std::endl;
	for(int i = 0; i< w*h; i++){
		sImgBuffer[i].r = yuv[i];
		sImgBuffer[i].g = 2*yuv[i] % 255;
		sImgBuffer[i].b = 3*yuv[i] % 255;
	}

	srcImg.createImageFromBuffer(w, h, PGM_DEPTH, sImgBuffer);

	*dstImg = new Image();
	int angle = rand()%360;
	rotation(srcImg, **dstImg, angle);
}
