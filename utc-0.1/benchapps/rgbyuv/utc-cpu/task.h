/*
 * task.h
 *
 *  Created on: Oct 9, 2017
 *      Author: Chao
 */

#include "Utc.h"
#include "image.h"

#ifndef BENCHAPPS_RGBYUV_UTC_CPU_TASK_H_
#define BENCHAPPS_RGBYUV_UTC_CPU_TASK_H_

class ImageCreate: public UserTaskBase{

public:

	void runImpl(Image* srcImg, char *infile_path);

};


class ImageOut: public UserTaskBase{
public:
	void runImpl(yuv_color_t* img, int w, int h, char *outfile_path);
};

class YUVconvertWorker : public UserTaskBase{
private:
	Image *srcImg;
	yuv_color_t *dstImg;

	PrivateScopedData<int> start_row;
	PrivateScopedData<int> end_row;
	PrivateScopedData<int> num_rows;
public:
	YUVconvertWorker():
		start_row(this),
		end_row(this),
		num_rows(this){

	}

	void initImpl(Image* srcImg, yuv_color_t *dstImg);

	void runImpl(double runtime[][1], int loop);
};



#endif /* BENCHAPPS_RGBYUV_UTC_CPU_TASK_H_ */
