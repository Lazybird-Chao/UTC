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
#include "CollectiveUtilities.h"

//G
#if ENABLE_SCOPED_DATA
#include "ScopedData/GlobalScopedData.h"
#include "ScopedData/GlobalScopedDataBase.h"
#endif

//I
#include "InprocConduit.h"

//L
#include "LockFreeRingbufferQueue.h"
#include "ProcList.h"

//P
#if ENABLE_SCOPED_DATA
#include "ScopedData/PrivateScopedDataBase.h"
#include "ScopedData/PrivateScopedData.h"
#include "ScopedData/internal_shmem/internal_win.h"
#endif

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

