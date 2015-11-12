#ifndef UTC_CONDUIT_H_
#define UTC_CONDUIT_H_

#include "TaskBase.h"
#include "ConduitBase.h"



namespace iUtc{

enum OpType{
	unknown =0,
	read,
	write,
	readby,
	writeby
};

// forward declaration
class ConduitManager;

class Conduit
{
public:
	Conduit();

	Conduit(TaskBase* srctask, TaskBase* dsttask);

	Conduit(TaskBase* srctask, TaskBase* dsttask, int capacity);

	//void setCapacity(int capacity);
	int getCapacity();

	std::string getName();

	TaskBase* getSrcTask();
	TaskBase* getDstTask();
	TaskBase* getAnotherTask();
	ConduitId getConduitId();

	void Connect(TaskBase* src, TaskBase* dst);

	int BWrite(void* DataPtr, int DataSize, int tag);
	int BWriteBy(ThreadRank thread, void* DataPtr, int DataSize, int tag);
	void BWriteBy_Finish(int tag);

	int Write(void* DataPtr, int DataSize, int tag);
	int WriteBy(ThreadRank thread, void* DataPtr, int DataSize, int tag);
	void WriteBy_Finish(int tag);

	int PWrite(void* DataPtr, int DataSize, int tag);
	int PWriteBy(ThreadRank thread, void* DataPtr, int DataSize, int tag);
	void PWriteBy_Finish(int tag);

	int Read(void* DataPtr, int DataSize, int tag);
	int ReadBy(ThreadRank thread, void* DataPtr, int DataSize, int tag);
	void ReadBy_Finish(int tag);

	~Conduit();

private:
	ConduitManager* m_cdtMgr;

	TaskBase* m_srcTask;
	TaskBase* m_dstTask;
	TaskId m_srcId;
	TaskId m_dstId;

	std::string m_Name;
	int m_conduitId;
	int m_capacity;

	//
	ConduitBase *m_realConduitPtr;


};


}// end namespace iUtc



#endif

