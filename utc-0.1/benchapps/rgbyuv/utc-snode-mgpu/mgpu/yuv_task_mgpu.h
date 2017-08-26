/*
 * yuv_task_sgpu.h
 *
 *  Created on: Mar 17, 2017
 *      Author: chao
 */

#ifndef YUV_TASK_SGPU_H_
#define YUV_TASK_SGPU_H_

#include "Utc.h"
#include "UtcGpu.h"

#include "../image.h"

using namespace iUtc;

class YUVconvertMGPU:public UserTaskBase{
private:
	Image *srcImg;
	yuv_color_t *dstImg;

	PrivateScopedData<int> start_row;
	PrivateScopedData<int> end_row;
	PrivateScopedData<int> num_rows;

public:
	YUVconvertMGPU()
		:start_row(this),
		 end_row(this),
		 num_rows(this){
	}
	void initImpl(Image* srcImg, yuv_color_t *dstImg);

	void runImpl(double runtime[][4], int loop, MemType memtype = MemType::pageable);

};



#endif /* YUV_TASK_SGPU_H_ */
