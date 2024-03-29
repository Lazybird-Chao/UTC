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



class RotateMGPU: public UserTaskBase{
private:
	Image *srcImg;
	Image *dstImg;
	int angle;

	PrivateScopedData<int> start_row;
	PrivateScopedData<int> end_row;
	PrivateScopedData<int> num_rows;

public:
	RotateMGPU():
		start_row(this),
		end_row(this),
		num_rows(this){

	}
	void initImpl(Image* srcImg, Image* dstImg, int angle);

	void runImpl(double runtime[][4], MemType memtype=MemType::pageable);
};




#endif /* ROTATE_TASK_H_ */
