/*
 * task.h
 *
 *  Created on: Oct 12, 2017
 *      Author: chaoliu
 */

#ifndef TASK_H_
#define TASK_H_

#include "Utc.h"


template<typename T>
class Output: public UserTaskBase{
public:
	void runImpl(T *buffer, int w, int h, char *ofile);
};


#endif /* TASK_H_ */
