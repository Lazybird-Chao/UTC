/*
 * yuv_task.h
 *
 *  Created on: Nov 14, 2017
 *      Author: chaoliu
 */

#ifndef YUV_TASK_H_
#define YUV_TASK_H_

#include "Utc.h"
#include <cstdint>
#include <vector>

class YUVconvertCPUWorker : public UserTaskBase{
private:
	int innerloop;
	int w;
	int h;
	int loop;

	uint32_t *srcImg_array;
	uint8_t *y_array;
	uint8_t *u_array;
	uint8_t *v_array;

	iUtc::Conduit *cdtIn;
	std::vector<iUtc::Conduit*> cdtOut;

	PrivateScopedData<int> start_row;
	PrivateScopedData<int> end_row;
	PrivateScopedData<int> num_rows;
public:
	YUVconvertCPUWorker():
		start_row(this),
		end_row(this),
		num_rows(this){

	}

	void initImpl(int w, int h, int innerloop, int loop,
			iUtc::Conduit *cdtIn, std::vector<iUtc::Conduit*> cdtOut);

	void runImpl(double runtime[][3]);
};



#endif /* YUV_TASK_H_ */
