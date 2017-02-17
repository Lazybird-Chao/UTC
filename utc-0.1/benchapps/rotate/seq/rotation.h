/*
 * rotation.h
 *
 *      Author: chao
 */

#ifndef ROTATION_H_
#define ROTATION_H_

#include "image.h"
#include "cmath"

#define PI 3.14159
#define PRECISION 3


void rotation(Image &In, Image &Out, int angle);

void filter(Pixel* colors, Pixel* dest, Coord* sample_pos);

inline void rotatePoint(Coord &pt, Coord &target, int angle){
	float rad = (float)angle/180 * PI;
	target.x = pt.x * cos(rad) - pt.y * sin(rad);
	target.y = pt.x * sin(rad) + pt.y * cos(rad);
}

/*
*	Function: round
*	---------------
*	Simple, optimized function for rounding fp numbers.
*/
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



#endif /* ROTATION_H_ */
