#ifndef UTC_CONDUIT_H_
#define UTC_CONDUIT_H_

#include "TaskBase.h"
#include "ConduitBase.h"

//#define ENABLE_OPBY_FINISH
#define DISABLE_OPBY_FINISH


namespace iUtc{

enum class ConduitType{
	unknown =0,
	c2c				//cputask to cputask
};

enum class ConduitOpType{
	unknown =0,
	read,
	write,
	readby,
	writeby,
	readbyfirst,
	writbyfirst
};

// forward declaration
class ConduitManager;

class Conduit
{
public:

	Conduit();

	Conduit(TaskBase* srctask, TaskBase* dsttask);

	//Conduit(TaskBase* srctask, TaskBase* dsttask, int capacity);

	//void setCapacity(int capacity);
	int getCapacity();

	std::string getName();

	TaskBase* getSrcTask();
	TaskBase* getDstTask();
	TaskBase* getAnotherTask();
	ConduitId_t getConduitId();

	void Connect(TaskBase* src, TaskBase* dst);

	int Write(void* DataPtr, DataSize_t DataSize, int tag);
	int WriteBy(ThreadRank_t thread, void* DataPtr, DataSize_t DataSize, int tag);
#ifdef ENABLE_OPBY_FINISH
	void WriteBy_Finish(int tag);
#endif
	int WriteByFirst(void* DataPtr, DataSize_t DataSize, int tag);

	int BWrite(void* DataPtr, DataSize_t DataSize, int tag);
	int BWriteBy(ThreadRank_t thread, void* DataPtr, DataSize_t DataSize, int tag);
#ifdef ENABLE_OPBY_FINISH
	void BWriteBy_Finish(int tag);
#endif

	int PWrite(void* DataPtr, DataSize_t DataSize, int tag);
	int PWriteBy(ThreadRank_t thread, void* DataPtr, DataSize_t DataSize, int tag);
#ifdef ENABLE_OPBY_FINISH
	void PWriteBy_Finish(int tag);
#endif

	int Read(void* DataPtr, DataSize_t DataSize, int tag);
	int ReadBy(ThreadRank_t thread, void* DataPtr, DataSize_t DataSize, int tag);
#ifdef ENABLE_OPBY_FINISH
	void ReadBy_Finish(int tag);
#endif
	int ReadByFirst(void* DataPtr, DataSize_t DataSize, int tag);


	int AsyncWrite(void* DataPtr, DataSize_t DataSize, int tag);
	void AsyncWrite_Finish(int tag);


	int AsyncRead(void* DataPtr, DataSize_t DataSize, int tag);
	void AsyncRead_Finish(int tag);

	~Conduit();

private:
	int initConduit_C2C(TaskBase* srctask, TaskBase* dsttask, int capacity);

	ConduitType		m_cdtType;
	ConduitManager* m_cdtMgr;

	TaskBase* m_srcTask;
	TaskBase* m_dstTask;
	TaskId_t m_srcId;
	TaskId_t m_dstId;

	std::string m_Name;
	int m_conduitId;
	int m_capacity;

	//
	ConduitBase *m_realConduitPtr;

	//
	Conduit& operator=(const Conduit& other)=delete;
	Conduit(const Conduit& other)=delete;


};


}// end namespace iUtc



#endif

