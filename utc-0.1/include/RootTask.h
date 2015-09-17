#ifndef UTC_ROOTTASK_H_
#define UTC_ROOTTASK_H_

#include "TaskBase.h"

#include <vector>

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

    protected:
        RootTask(); //default constructor is disabled


    };//class RootTask

}// namespace iUtc




#endif
