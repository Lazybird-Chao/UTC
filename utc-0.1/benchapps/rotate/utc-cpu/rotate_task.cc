/*
 * rotate_task.cc
 *
 *  Created on: Oct 8, 2017
 *      Author: Chao
 */

#include "task.h"
#include "image.h"
#include <cmath>

#define PI 3.14159
#define PRECISION 3



inline void rotatePoint(Coord &pt, Coord &target, int angle){
	float rad = (float)angle/180 * PI;
	target.x = pt.x * cos(rad) - pt.y * sin(rad);
	target.y = pt.x * sin(rad) + pt.y * cos(rad);
}

inline double myround(double num, int digits) {
    double v[] = {1, 10, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8};
	if(digits > (sizeof(v)/sizeof(double))) return num;
    return floor(num * v[digits] + 0.5) / v[digits];
}

/*
*	Function: interpolateLinear
*	---------------------------
*	Linearly interpolates two pixel colors according to a given weight factor.
*/
inline void interpolateLinear(Pixel* a, Pixel* b, Pixel* dest, float weight) {
	dest->r = a->r * (1.0-weight) + b->r * weight;
	dest->g = a->g * (1.0-weight) + b->g * weight;
	dest->b = a->b * (1.0-weight) + b->b * weight;
}

/*
*	Function: filter
*	---------------------
*	Filters a given array of pixel colours of length 4, blending
*	color values into a final pixel. The algorithm used is bilinear
*	filtering, using the sample position as a weight for color blend.
*/
void filter(Pixel* colors, Pixel* dest, Coord* sample_pos) {
	//uint32_t r, g, b;
	Pixel sample_v_upper, sample_v_lower;
	float x_weight = myround(sample_pos->x - floor(sample_pos->x), PRECISION);
	float y_weight = myround(sample_pos->y - floor(sample_pos->y), PRECISION);

	interpolateLinear(&colors[0], &colors[3], &sample_v_upper, x_weight);
	interpolateLinear(&colors[1], &colors[2], &sample_v_lower, x_weight);
	interpolateLinear(&sample_v_upper, &sample_v_lower, dest, y_weight);
}


void RotateWorker::initImpl(Image* srcImg, Image* dstImg, int angle){
	if(__localThreadId == 0){
		this->srcImg = srcImg;
		this->dstImg = dstImg;
		this->angle = angle;

		/*
		 * compute the out image's size
		 */
		Coord ul, ur, ll, lr;
		float xc = (float)srcImg->getWidth() / 2.0;
		float yc = (float)srcImg->getHeight() / 2.0;
		ul.x = -xc;
		ul.y = yc;
		ur.x = xc;
		ur.y = yc;
		ll.x = -xc;
		ll.y = -yc;
		lr.x = xc;
		lr.y = -yc;
		Coord outCorner[4];
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
		dstImg->createImageFromTemplate(outW, outH, srcImg->getDepth());
	}
	__fastIntraSync.wait();
	int rowsPerThread = dstImg->getHeight()/__numLocalThreads;
	int residue = dstImg->getHeight() % __numLocalThreads;
	if(__localThreadId < residue){
		local_startRow = (rowsPerThread+1)*__localThreadId;
		local_endRow = local_startRow + rowsPerThread;
	}else{
		local_startRow = rowsPerThread*__localThreadId + residue;
		local_endRow = local_startRow + rowsPerThread-1;
	}

	intra_Barrier();
	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<" finish initImpl.\n";
	}
}


void RotateWorker::runImpl(double runtime[][1]){
	if(__localThreadId == 0){
		std::cout<<getCurrentTask()->getName()<<" begin run ..."<<std::endl;
	}

	Timer timer, timer0;
	timer.start();
	int outW = dstImg->getWidth();
	int outH = dstImg->getHeight();
	int inW = srcImg->getWidth();
	int inH = srcImg->getHeight();
	float xc = (float)inW/ 2.0;
	float yc = (float)inH / 2.0;
	int rev_angle = 360 - angle;
	float x_offset_target = (float)outW/2.0;
	float y_offset_target = (float)outH/2.0;
	//std::cout<<local_startRow.load()<<" "<<local_endRow.load()<<std::endl;
	//sleep(__localThreadId);
	//__fastIntraSync.wait();
	for(int i = local_startRow; i <= local_endRow; i++){
		for(int j = 0; j<outW; j++){
			/* Find origin pixel for current destination pixel */
			Coord cur = {-x_offset_target + (float)j, y_offset_target - (float)i};
			Coord origin_pix;
			rotatePoint(cur, origin_pix,rev_angle);

			/* If original image contains point, sample colour and write back */
			if(srcImg->containsPixel(&origin_pix)) {
				int samples[4][2];
				Pixel colors[4];

				/* Get sample positions */
				for(int k = 0; k < 4; k++) {
					samples[k][0] = (int)(origin_pix.x + xc) + ((k == 2 || k == 3) ? 1 : 0);
					samples[k][1] = (int)(-origin_pix.y + yc) + ((k == 1 || k == 3) ? 1 : 0);
					// Must make sure sample positions are still valid image pixels
					if(samples[k][0] >= inW)
						samples[k][0] = inW-1;
					if(samples[k][1] >= inH)
						samples[k][1] = inH-1;
				}

				/* Get colors for samples */
				for(int k = 0; k < 4; k++) {
					colors[k] = srcImg->getPixelAt(samples[k][0], samples[k][1]);
				}

				/* Filter colors */
				Pixel final;
				filter(colors, &final, &origin_pix);

				/* Write output */
				dstImg->setPixelAt(j, i, &final);
			} else {
				/* Pixel is not in source image, write black color */
				Pixel final = {0,0,0};
				dstImg->setPixelAt(j, i, &final);
			}
		}
	}
	//std::cout<<ERROR_LINE<<std::endl;
	__fastIntraSync.wait();
	double totaltime = timer.stop();

	runtime[__localThreadId][0] = totaltime;
	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<" finish runImpl.\n";
	}

}


