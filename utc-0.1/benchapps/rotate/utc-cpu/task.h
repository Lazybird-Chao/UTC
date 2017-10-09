/*
 * task.h
 *
 *  Created on: Mar 17, 2017
 *      Author: chao
 */

#ifndef TASK_H_
#define TASK_H_

#include "Utc.h"

#include "image.h"

class ImageCreate: public UserTaskBase{
private:


public:

	void runImpl(Image* srcImg, char *infile_path);

};


class ImageOut: public UserTaskBase{
public:
	void runImpl(Image* img, char *outfile_path);
};


class RotateWorker: public UserTaskBase{
private:
	Image *srcImg;
	Image *dstImg;
	int angle;

	PrivateScopedData<int> local_startRow;
	PrivateScopedData<int> local_endRow;

public:
	void initImpl(Image* srcImg, Image* dstImg, int angle);
	void runImpl(double runtime[][1]);

	RotateWorker():
		local_startRow(this),
		local_endRow(this){

	}

};


#endif /* TASK_H_ */
