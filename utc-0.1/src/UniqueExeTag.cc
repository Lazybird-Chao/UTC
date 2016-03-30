/*
 * UniqueExeTag.cc
 *
 *  Created on: Mar 29, 2016
 *      Author: chao
 */

#include "UniqueExeTag.h"

namespace iUtc{

UniqueExeTag::UniqueExeTag(int nthreads, int ntags){
	m_uniqueExeTag = new std::atomic<bool>[ntags];
	for(int i=0; i<ntags; i++)
		m_uniqueExeTag[i]=true;
	m_uniqueExeIdx = new int[nthreads];
	for(int i=0; i<nthreads; i++)
		m_uniqueExeIdx[i]=0;
	m_nthreads = nthreads;
	m_ntags= ntags;
}

void UniqueExeTag::reset(){
	for(int i=0; i< m_ntags; i++)
		m_uniqueExeTag[i] = true;
	for(int i=0; i<m_nthreads; i++)
		m_uniqueExeIdx[i] = 0;
}

bool UniqueExeTag::getUniqueExe(int tid){
	bool tag = true;
	if(m_uniqueExeTag[m_uniqueExeIdx[tid]].compare_exchange_strong(tag, false)){
		m_uniqueExeIdx[tid]++;
		return true;
	}
	m_uniqueExeIdx[tid]++;
	return false;
}

}// end namespace iUtc


