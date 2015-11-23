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
	ConduitId_t getConduitId();

	void Connect(TaskBase* src, TaskBase* dst);

	int BWrite(void* DataPtr, DataSize_t DataSize, int tag);
	int BWriteBy(ThreadRank_t thread, void* DataPtr, DataSize_t DataSize, int tag);
	void BWriteBy_Finish(int tag);

	int Write(void* DataPtr, DataSize_t DataSize, int tag);
	int WriteBy(ThreadRank_t thread, void* DataPtr, DataSize_t DataSize, int tag);
	void WriteBy_Finish(int tag);

	int PWrite(void* DataPtr, DataSize_t DataSize, int tag);
	int PWriteBy(ThreadRank_t thread, void* DataPtr, DataSize_t DataSize, int tag);
	void PWriteBy_Finish(int tag);

	int Read(void* DataPtr, DataSize_t DataSize, int tag);
	int ReadBy(ThreadRank_t thread, void* DataPtr, DataSize_t DataSize, int tag);
	void ReadBy_Finish(int tag);

	int AsyncWrite(void* DataPtr, DataSize_t DataSize, int tag);
	void AsyncWrite_Finish(int tag);


	int AsyncRead(void* DataPtr, DataSize_t DataSize, int tag);
	void AsyncRead_Finish(int tag);

	~Conduit();

private:
	ConduitManager* m_cdtMgr;

	TaskBase* m_srcTask;
	TaskBase* m_dstTask;
	TaskId_t m_srcId;
	TaskId_t m_dstId;

	std::string m_Name;
	int m_conduitId;
	// used for inproc-conduit, the number of available inner buffer items
	int m_capacity;

	//
	ConduitBase *m_realConduitPtr;

	//
	Conduit& operator=(const Conduit& other)=delete;
	Conduit(const Conduit& other)=delete;


};


}// end namespace iUtc



#endif

