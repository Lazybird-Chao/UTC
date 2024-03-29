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

#include "../image.h"

using namespace iUtc;



class RotateSGPU: public UserTaskBase{
private:
	Image *srcImg;
	Image *dstImg;
	int angle;

public:
	void initImpl(Image* srcImg, Image* dstImg, int angle);

	void runImpl(double *runtime, MemType memtype=MemType::pageable);
};




#endif /* ROTATE_TASK_H_ */
