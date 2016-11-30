#ifndef UTC_H_
#define UTC_H_

//TOP
#include "UtcBasics.h"

//A
#include "AffinityUtilities.h"

//B
#include "Barrier.h"

//C
#include "ConduitManager.h"
#include "ConduitBase.h"
#include "Conduit.h"

//G
#include "GlobalScopedData.h"
#include "GlobalScopedDataBase.h"

//I
#include "InprocConduit.h"

//L
#include "LockFreeRingbufferQueue.h"
#include "ProcList.h"

//P
#include "PrivateScopedDataBase.h"
#include "PrivateScopedData.h"

//R
#include "RootTask.h"

//S
#include "SharedDataLock.h"
#include "SpinBarrier.h"
#include "SpinLock.h"

//T
#include "TaskBase.h"
#include "TaskInfo.h"
#include "TaskManager.h"
#include "Task.h"
#include "TaskArtifact.h"
#include "TaskCPU.h"
#include "TaskUtilities.h"
#include "Timer.h"
#include "TimerUtilities.h"


//U
#include "UtcBase.h"
#include "UtcBasics.h"
#include "UtcContext.h"
#include "UtcException.h"
#include "UtcMpi.h"
#include "UserTaskBase.h"

//X
#include "XprocConduit.h"

//Y

//Z


/****** gpu task support ******/
#include "gpu/UtcGpuBasics.h"
#if ENABLE_GPU_TASK

#include "gpu/UtcGpu.h"

#endif


#endif

