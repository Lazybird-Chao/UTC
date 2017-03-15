/*
 * rotate_task.h
 *
 *  Created on: Mar 15, 2017
 *      Author: chao
 */

#ifndef ROTATE_TASK_H_
#define ROTATE_TASK_H_

#include "Utc.h"
#include "UtcGpu.h"

#include "image.h"

using namespace iUtc;

class ImageCreate: public UserTaskBase{
private:


public:

	void runImpl(Image* srcImg, char *infile_path);

};


class ImageOut: public UserTaskBase{
public:
	void runImpl(Image* img, char *outfile_path);
};



class Rotate: public UserTaskBase{
private:
	Image *srcImg;
	Image *dstImg;

public:
	void initImpl(Image* srcImg, Image* dstImg);

	void runImpl(int angle, double *runtime, MemType memtype=MemType::pageable);
};




#endif /* ROTATE_TASK_H_ */
