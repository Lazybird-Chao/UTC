#ifndef UTC_CONDUITBASE_H_
#define UTC_CONDUITBASE_H_

#include "UtcBasics.h"

namespace iUtc{

class ConduitBase
{

public:

	ConduitBase(){}

	virtual int BWrite(void* DataPtr, DataSize_t DataSize, int tag)=0;
	virtual int BWriteBy(ThreadRank_t thread, void* DataPtr, DataSize_t DataSize, int tag)=0;
	virtual void BWriteBy_Finish(int tag)=0;

	virtual int Write(void* DataPtr, DataSize_t DataSize, int tag)=0;
	virtual int WriteBy(ThreadRank_t thread, void* DataPtr, DataSize_t DataSize, int tag)=0;
	virtual void WriteBy_Finish(int tag)=0;

	virtual int PWrite(void* DataPtr, DataSize_t DataSize, int tag)=0;
	virtual int PWriteBy(ThreadRank_t thread, void* DataPtr, DataSize_t DataSize, int tag)=0;
	virtual void PWriteBy_Finish(int tag)=0;


	virtual int Read(void* DataPtr, DataSize_t DataSize, int tag)=0;
	virtual int ReadBy(ThreadRank_t thread, void* DataPtr, DataSize_t DataSize, int tag)=0;
	virtual void ReadBy_Finish(int tag)=0;

	virtual int AsyncWrite(void* DataPtr, DataSize_t DataSize, int tag)=0;
	virtual void AsyncWrite_Finish(int tag)=0;


	virtual int AsyncRead(void* DataPtr, DataSize_t DataSize, int tag)=0;
	virtual void AsyncRead_Finish(int tag)=0;


	virtual ~ConduitBase(){}


};



}// end namespace iUtc




#endif
