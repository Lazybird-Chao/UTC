/*
 * rotate_task.h
 *
 *  Created on: Nov 14, 2017
 *      Author: chaoliu
 */

#ifndef ROTATE_TASK_H_
#define ROTATE_TASK_H_

#include "image.h"
#include "Utc.h"

class RotateCPUWorker: public UserTaskBase{
private:
	int w;
	int h;

	int outW;
	int outH;

	Image srcImg;
	Image dstImg;
	uint8_t *yuv;
	int angle;

	iUtc::Conduit *cdtIn;
	iUtc::Conduit *cdtOut;

	PrivateScopedData<int> local_startRow;
	PrivateScopedData<int> local_endRow;

public:
	void initImpl(int w, int h, iUtc::Conduit *cdtIn, iUtc::Conduit *cdtOut);
	void runImpl(double runtime[][3], int loop);

	RotateCPUWorker():
		local_startRow(this),
		local_endRow(this){

	}

};


#endif /* ROTATE_TASK_H_ */
