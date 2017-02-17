/*
 * image.h
 *
 *      Author: chao
 */

#ifndef IMAGE_H_
#define IMAGE_H_

#include <fstream>
#include <iostream>
#include <stdint.h>
#include <stdio.h>

#define RGB_DEPTH 3
#define PGM_DEPTH 1
#define RGB_MAX_COLOR 255

/*
*	Structure: Pixel
*	-------------
*	Structure representing the color of an RGB image pixel.
*/
typedef struct __attribute__((aligned(4))){
	uint8_t r, g, b;
} Pixel;

/*
*	Structure: Coord
*	-------------
*	Structure for representation of 2-dimensional coordinates.
*/
typedef struct {
	float x;
	float y;
} Coord;

/*
*	Class: Image
*	------------
*	Class representing an RGB image. Knows image size, color depth
*   and contains the RGB color values of the image.
*/
class Image {
	public:
        void createImageFromBuffer(int width, int height, int depth, Pixel* pels);
		bool createImageFromFile(const char *fname);
		void createImageFromTemplate(int width, int height, int depth);
		Pixel getPixelAt(int x, int y);
		void setPixelAt(int x, int y, Pixel* p);
		bool containsPixel(Coord* pix);
		int getWidth();
        int getHeight();
        int getDepth();
        int getMaxcolor();
		void clean();
		Pixel *getPixelBuffer();
	private:
		Pixel* pixels;
		int width, height;
		int depth, maxcolor;
		float x_off, y_off;
		int ppmGetInt(std::fstream &src);
		char ppmGetChar(std::fstream &src);
};

#endif /* IMAGE_H_ */
