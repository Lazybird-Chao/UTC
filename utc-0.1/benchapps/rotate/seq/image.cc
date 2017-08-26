/*
 * image.cc
 *
 *      Author: chao
 */

#include "image.h"

/*
*   Function: createImageFromFile
*   -----------------------------
*   Tries to open the file specified by the given file name
*   and, if successful, fills the Image object with resolution and
*   depth information and fills the pixel vector.
*/
bool Image::createImageFromFile(const char* fname) {
    std::fstream in;
    char buf[2];

    in.open(fname);
    if(!in.is_open()) {
    	std::cerr << "Cannot Open File " << fname << std::endl;
        return false;
    }
    /* Make sure it is an RGB binary file */
	in.read(buf, 2);
    if((buf[0] != 'P') || ((buf[1] != '6') && (buf[1] != '5'))) {
        std::cerr << "Wrong Image File Format: " << buf[0] << buf[1] << std::endl;
        in.close();
        return false;
    }
    if(buf[1] == '5') {
        depth = PGM_DEPTH;
        std::cerr << "Grayscale Currently Not Supported" << std::endl;
        in.close();
        return false;
    }
    else
        depth = RGB_DEPTH;

    width = ppmGetInt(in);
    height = ppmGetInt(in);
    maxcolor = ppmGetInt(in);
	x_off = (float)width / 2.0;
	y_off = (float)height / 2.0;
    pixels = new Pixel[width*height];
    for(int i = 0; i < width * height; i++) {
        in.read((char*)&pixels[i], 3);
    }
	in.close();
    return true;
}

/*
*   Function: createImageFromBuffer
*   -------------------------------
*   Creates an Image object from a given buffer of pixels.
*   To accomplish this, image size information must be passed along.
*/
void Image::createImageFromBuffer(int width, int height, int depth, Pixel* pels) {
    this->width = width;
    this->height = height;
    this->depth = depth;
    this->maxcolor = RGB_MAX_COLOR;
    pixels = new Pixel[width*height];
    for(int i = 0; i < width*height; i++)
        pixels[i] = pels[i];
    return;
}

/*
*	Function: createImageFromTemplate
*	---------------------------------
*	Creates an "empty" image whose contents can be filled by using
*	the setPixelAt function.
*/
void Image::createImageFromTemplate(int width, int height, int depth) {
	this->width = width;
	this->height = height;
	this->depth = depth;
	this->maxcolor = RGB_MAX_COLOR;
	x_off = (float)width / 2.0;
	y_off = (float)height / 2.0;
    pixels = new Pixel[width*height];
    for(int i = 0; i < width * height; i++) {
        pixels[i].r = 0;
        pixels[i].g = 0;
        pixels[i].b = 0;
    }
}


int Image::getWidth() {
    return width;
}

int Image::getHeight() {
    return height;
}

int Image::getDepth() {
    return depth;
}

int Image::getMaxcolor() {
    return maxcolor;
}

Pixel Image::getPixelAt(int x, int y) {
    return pixels[y*width+x];
}

void Image::setPixelAt(int x, int y, Pixel* p) {
    if(x < width && y < height) {
        pixels[width*y + x].r = p->r;
        pixels[width*y + x].g = p->g;
        pixels[width*y + x].b = p->b;
    }
}

/*
*	Function: containsPixel
*	-----------------------
*	Returns true if the supplied coordinates lie within the picture
*	or at maximum one pixel outside the picture.
*/
bool Image::containsPixel(Coord* pix) {
	bool x_correct = (pix->x < x_off) && (pix->x > (0.0-x_off));
	bool y_correct = (pix->y < y_off) && (pix->y > (0.0-y_off));
	if(x_correct && y_correct)
		return true;
	return false;
}

void Image::clean() {
    if(pixels) {
        delete[] pixels;
        pixels = NULL;
    }
}

Pixel *Image::getPixelBuffer(){
	return pixels;
}


/*
*   Function: ppmGetInt
*   -------------------
*   Extracts the next integer value from the source file stream.
*   Used for getting the width, height and maxcolor values of the
*   source image.
*/
int Image::ppmGetInt(std::fstream &src) {
    char ch;
	int i;

	do {
		ch = ppmGetChar(src);
	} while (ch==' ' || ch=='\t' || ch=='\n' || ch=='\r' || ch=='\x00A');

	i = 0;

	do {
		i = i*10 + ch-'0';
		ch = ppmGetChar(src);
	} while (ch>='0' && ch<='9');

	return i;
}

/*
*	Function: ppmGetChar
*	--------------------
*	Returns the next char from the fstream src,
*	ignores .ppm user comments
*/
char Image::ppmGetChar(std::fstream &src) {
   char ch;

   ch = (char)src.get();

   if (ch=='#')
   {
      do {
         ch = (char)src.get();
      } while (ch!='\n' && ch!='\r');
   }

   return ch;
}


