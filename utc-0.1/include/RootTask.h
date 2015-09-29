#ifndef UTC_ROOTTASK_H_
#define UTC_ROOTTASK_H_

#include "TaskBase.h"

#include <vector>
#include <fstream>

namespace iUtc{

/**
 * RootTask is a container of the information for the main SPMD Task
 * This is not meant to be instantiated directly by the user
 */
    class RootTask: public TaskBase
    {
    public:

        //worldsize = the number of processes in the world
        RootTask(int WorldSize, int currentProcess);

        ~RootTask();

        std::ofstream* getProcOstream();
        void setProcOstream(std::ofstream& procOstream);

    protected:
        RootTask(); //default constructor is disabled

        //std::ofstream *m_procOstream;  move to taskbase

    };//class RootTask

}// namespace iUtc




#endif
