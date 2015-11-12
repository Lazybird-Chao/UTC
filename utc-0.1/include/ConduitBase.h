#ifndef UTC_CONDUITBASE_H_
#define UTC_CONDUITBASE_H_

#include "UtcBasics.h"

namespace iUtc{

class ConduitBase
{

public:

	ConduitBase(){}

	virtual int BWrite(void* DataPtr, int DataSize, int tag)=0;
	virtual int BWriteBy(ThreadRank thread, void* DataPtr, int DataSize, int tag)=0;
	virtual void BWriteBy_Finish(int tag)=0;

	virtual int Write(void* DataPtr, int DataSize, int tag)=0;
	virtual int WriteBy(ThreadRank thread, void* DataPtr, int DataSize, int tag)=0;
	virtual void WriteBy_Finish(int tag)=0;

	virtual int PWrite(void* DataPtr, int DataSize, int tag)=0;
	virtual int PWriteBy(ThreadRank thread, void* DataPtr, int DataSize, int tag)=0;
	virtual void PWriteBy_Finish(int tag)=0;


	virtual int Read(void* DataPtr, int DataSize, int tag)=0;
	virtual int ReadBy(ThreadRank thread, void* DataPtr, int DataSize, int tag)=0;
	virtual void ReadBy_Finish(int tag)=0;

	virtual ~ConduitBase(){}


};



}// end namespace iUtc




#endif
