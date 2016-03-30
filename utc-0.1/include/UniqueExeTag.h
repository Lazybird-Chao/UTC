/*
 * UniqueExeTag.h
 *
 *  Created on: Mar 29, 2016
 *      Author: chao
 */

#ifndef INCLUDE_UNIQUEEXETAG_H_
#define INCLUDE_UNIQUEEXETAG_H_

#include <atomic>

namespace iUtc{


//
class UniqueExeTag{
public:
	void UniqueExeTag(int nthreads, int ntags);

	void reset();

	bool getUniqueExe(int tid);

private:
	std::atomic<bool> *m_uniqueExeTag;
	int *m_uniqueExeIdx;
	int m_nthreads;
	int m_ntags;

};

}//end namespace iUtc

#endif /* INCLUDE_UNIQUEEXETAG_H_ */
