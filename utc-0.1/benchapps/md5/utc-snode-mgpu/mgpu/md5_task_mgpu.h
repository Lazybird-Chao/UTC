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

class MD5MGPU: public UserTaskBase{
private:
	config_t *md5Config;

	static thread_local long local_numBuffers;
	static thread_local long local_startBufferIndex;
	static thread_local uint8_t* local_buffer;

public:
	void initImpl(config_t *args);

	void runImpl(double runtime[][4], int blocksize, MemType memtype);

};



#endif /* MD5_TASK_SGPU_H_ */
