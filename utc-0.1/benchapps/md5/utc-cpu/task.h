/*
 * task.h
 *
 *  Created on: Oct 10, 2017
 *      Author: chaoliu
 */

#ifndef TASK_H_
#define TASK_H_

#include "md5.h"
#include "Utc.h"

class RandomInput : public UserTaskBase{
public:
	void runImpl(config_t *configArgs, const char* filename, bool isBinary);

};

class Output : public UserTaskBase{
public:
	void runImpl(config_t *args);
};

class MD5Worker : public UserTaskBase{
private:
	config_t *md5Config;
	static thread_local long local_numBuffers;
	static thread_local long local_startBufferIndex;
public:
	void initImpl(config_t *args);

	void runImpl(double runtime[][1]);

	void process(uint8_t *in, uint8_t *out, long bufsize);
};

void toFile(char* data, long numBuffs, long buffSize, const char* filename, bool isBinary);

void fromFile(char* &data, long& numBuffs, long& buffSize, const char* filename, bool isBinary);

void increaseBy(int times, config_t *configArgs);


#endif /* TASK_H_ */
