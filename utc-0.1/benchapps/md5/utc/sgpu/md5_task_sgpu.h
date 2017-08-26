/*
 * md5_task_sgpu.h
 *
 *  Created on: Mar 22, 2017
 *      Author: chao
 */

#ifndef MD5_TASK_SGPU_H_
#define MD5_TASK_SGPU_H_

#include "Utc.h"
#include "UtcGpu.h"
#include "../md5.h"

using namespace iUtc;

class MD5SGPU: public UserTaskBase{
private:
	config_t *md5Config;

public:
	void initImpl(config_t *args);

	void runImpl(double* runtime, int blocksize, MemType memtype);

};



#endif /* MD5_TASK_SGPU_H_ */
