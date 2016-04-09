/*
 * UniqueExeTag.cc
 *
 *  Created on: Mar 29, 2016
 *      Author: chao
 */

#include "UniqueExeTag.h"

namespace iUtc{

UniqueExeTag::UniqueExeTag(int nthreads, int ntags){
	m_uniqueExeTag = new std::atomic<int>[ntags];
	for(int i=0; i<ntags; i++)
		m_uniqueExeTag[i]=0;
	m_uniqueExeIdx = new int[nthreads];
	for(int i=0; i<nthreads; i++)
		m_uniqueExeIdx[i]=0;
	m_nthreads = nthreads;
	m_ntags= ntags;
}

void UniqueExeTag::reset(){
	for(int i=0; i< m_ntags; i++)
		m_uniqueExeTag[i] = 0;
	for(int i=0; i<m_nthreads; i++)
		m_uniqueExeIdx[i] = 0;
}

bool UniqueExeTag::getUniqueExe(int tid){
	if(m_nthreads==1)
		return true;
	int tag = 0;
	if(m_uniqueExeTag[m_uniqueExeIdx[tid]].compare_exchange_strong(tag, 1)){
		m_uniqueExeIdx[tid]++;
		m_uniqueExeIdx[tid] = m_uniqueExeIdx[tid]%m_ntags;
		return true;
	}
	else{
		while(1){
			int oldvalue = m_uniqueExeTag[m_uniqueExeIdx[tid]].load();
			if(oldvalue == m_nthreads-1){
				m_uniqueExeTag[m_uniqueExeIdx[tid]].store(0);
				break;
			}
			if(m_uniqueExeTag[m_uniqueExeIdx[tid]].compare_exchange_strong(oldvalue, oldvalue+1))
				break;
		}
		m_uniqueExeIdx[tid]++;
		m_uniqueExeIdx[tid] = m_uniqueExeIdx[tid]%m_ntags;
		return false;
	}
}

UniqueExeTag::~UniqueExeTag(){
	if(m_uniqueExeTag)
		delete m_uniqueExeTag;
	if(m_uniqueExeIdx)
		delete m_uniqueExeIdx;
}

}// end namespace iUtc


