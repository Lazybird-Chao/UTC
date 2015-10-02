#include "Conduit.h"
#include "ConduitManager.h"

#include <cstdlib>


namespace iUtc
{

Conduit::Conduit()
{
    m_srcTask = nullptr;
    m_dstTask = nullptr;
    m_srcId = -1;
    m_dstId = -1;

    m_conduitId = ConduitManager::getNewConduitId();
    m_capacity = CONDUIT_CAPACITY_DEFAULT;
    m_availableMsgBuffHead = 1;
    m_availableMsgBuffTail = m_capacity;

    m_msgBuffPool.clear();

    ConduitManager* cdtMgr = ConduitManager::getInstance();
    cdtMgr->registerConduit(this);

}

Conduit::Conduit(TaskBase* srctask, TaskBase* dsttask)
{
    m_srcTask = srctask;
    m_dstTask = dsttask;
    m_srcId = m_srcTask->getTaskId();
    m_dstId = m_dstTask->getTaskId();

    m_conduitId = ConduitManager::getNewConduitId();
    m_capacity = CONDUIT_CAPACITY_DEFAULT;
    m_availableMsgBuffHead = 1;
    m_availableMsgBuffTail = m_capacity;

    m_msgBuffPool.clear();
    ConduitManager* cdtMgr = ConduitManager::getInstance();
    cdtMgr->registerConduit(this);

}

void Conduit::setCapacity(int capacity)
{
    m_capacity = capacity;
}

int Conduit::getCapacity()
{
    return m_capacity;
}

TaskBase* Conduit::getSrcTask()
{
    return m_srcTask;
}

TaskBase* Conduit::getDstTask()
{
    return m_dstTask;
}

void Conduit::Connect(TaskBase* src, TaskBase* dst)
{
    if(m_srcTask || m_dstTask)
    {
        std::cout<<"Error, already connected\n";
        exit(1);
    }
    m_srcTask = src;
    m_dstTask = dst;
    return;
}

ConduitId Conduit::getConduitId()
{
    return m_conduitId;
}

int ConduitWrite(void* DataPtr, int DataSize, int DataTag)
{
    // one task thread get write lock
    std::unique_lock<std::mutex> LCK1(m_srcMsgBuffMutex);
    if(m_srcMsgBuffPool.find(DataTag)!=m_srcMsgBuffPool.end())
    {
        // message tag already exist, when one thread see tag exist, there may
        // have two cases:
        //   1: all threads call this function, only one thread get the lock,
        //       do data write and record tag; so when other threads get the
        //       lock, they would see the tag exist, so just return, this is
        //        what we want
        //   2: the tag is used again before the tag is released. so even the
        //       first coming thread would see the tag exist. Here is an program
        //        error. But we can't distinguish these two case now, we just
        //       ignore error. so user should ensure that not to use same tag
        //       more than once, before it's released.
        // TODO: need add more info to check tag reuse error!
        return 0;
    }
    void *tmp_buffer= malloc(DataSize);
    m_srcMsgBuffPool.insert(std::<MessageTag,)
}




}




