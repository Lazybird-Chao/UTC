/*
 * output_task.h
 *
 *  Created on: Nov 14, 2017
 *      Author: chaoliu
 */

#ifndef OUTPUT_TASK_H_
#define OUTPUT_TASK_H_

#include "Utc.h"
#include "image.h"

class OutputWorker: public UserTaskBase{
public:
	void runImpl(int loop, iUtc::Conduit *cdtIn, double runtime[][3], int id);
};



#endif /* OUTPUT_TASK_H_ */
