/*
 * task.h
 *
 *  Created on: Mar 17, 2017
 *      Author: chao
 */

#ifndef TASK_H_
#define TASK_H_

#include "Utc.h"
#include "UtcGpu.h"

#include "image.h"

class ImageCreate: public UserTaskBase{
private:


public:

	void runImpl(Image* srcImg, char *infile_path);

};


class ImageOut: public UserTaskBase{
public:
	void runImpl(yuv_color_t* img, int w, int h, char *outfile_path);
};


#endif /* TASK_H_ */
