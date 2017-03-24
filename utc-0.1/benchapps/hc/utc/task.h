/*
 * task.h
 *
 *  Created on: Mar 23, 2017
 *      Author: Chao
 */

#ifndef BENCHAPPS_HC_UTC_TASK_H_
#define BENCHAPPS_HC_UTC_TASK_H_

#include "Utc.h"
#include <cstdlib>
#include <cstdio>

template<typename T>
class Output: public UserTaskBase{
public:
	void runImpl(T *buffer, int w, int h, char *ofile){
		FILE *fp = fopen(ofile, "w");
		for(int i=0; i<h; i++){
			for(int j=0; j<w; j++){
				fprintf(fp, "%.5f ", buffer[i*w +j]);
			}
			fprintf(fp,"\n");
		}
		fclose(fp);
	}
};




#endif /* BENCHAPPS_HC_UTC_TASK_H_ */
