/*
 * rotation.cc
 *
 *      Author: chao
 */

#include "image.h"
#include "rotation.h"
#include <cfloat>

/*
 * take a image object as in, rotate angle degree, and write the result
 * in out image object
 */
void rotation(Image &In, Image &Out, int angle){

	int height = In.getHeight();
	int width = In.getWidth();
	int depth = In.getDepth();

	if(angle == 0 || angle == 360){
		Out.createImageFromBuffer(width, height, depth, In.getPixelBuffer());
		return;
	}

	/* Steps for rotation:
		1. Determine target image size by rotating corners
		2. For each pixel in target image, do
			- backwards rotation to determine origin location
			- for origin location, sample and filter 4 closest neighbour pixels
			- write colour value appropriately
	*/
	Coord ul, ur, ll, lr;
	float xc = (float)width / 2.0;
	float yc = (float)height / 2.0;
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
	Out.createImageFromTemplate(outW, outH, depth);

	int rev_angle = 360 - angle;
	float x_offset_target = (float)outW/2.0;
	float y_offset_target = (float)outH/2.0;
	for(int i = 0; i < outH; i++) {
		for(int j = 0; j < outW; j++) {
			/* Find origin pixel for current destination pixel */
			Coord cur = {-x_offset_target + (float)j, y_offset_target - (float)i};
			Coord origin_pix;
			rotatePoint(cur, origin_pix,rev_angle);

			/* If original image contains point, sample colour and write back */
			if(In.containsPixel(&origin_pix)) {
				int samples[4][2];
				Pixel colors[4];

				/* Get sample positions */
				for(int k = 0; k < 4; k++) {
					samples[k][0] = (int)(origin_pix.x + xc) + ((k == 2 || k == 3) ? 1 : 0);
					samples[k][1] = (int)(-origin_pix.y + yc) + ((k == 1 || k == 3) ? 1 : 0);
					// Must make sure sample positions are still valid image pixels
					if(samples[k][0] >= width)
						samples[k][0] = width-1;
					if(samples[k][1] >= height)
						samples[k][1] = height-1;
				}

				/* Get colors for samples */
				for(int k = 0; k < 4; k++) {
					colors[k] = In.getPixelAt(samples[k][0], samples[k][1]);
				}

				/* Filter colors */
				Pixel final;
				filter(colors, &final, &origin_pix);

				/* Write output */
				Out.setPixelAt(j, i, &final);
			} else {
				/* Pixel is not in source image, write black color */
				Pixel final = {0,0,0};
				Out.setPixelAt(j, i, &final);
			}
		}
	}
	return;

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




