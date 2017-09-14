#####################
default:libutc.a

include ../for-dis.comm

#####################

vpath %.h  $(PROJECT_INCLUDEDIR)
vpath %.inc $(PROJECT_INCLUDEDIR)
vpath %.cc $(PROJECT_SRCDIR)


###################
OBJS:= 	UtcContext.o \
		UtcMpi.o \
		ProcList.o \
		UniqueExeTag.o \
    	RootTask.o  \
    	TaskBase.o  \
    	TaskManager.o \
    	Task.o  \
    	TaskCPU.o \
    	TaskUtilities.o \
    	UserTaskBase.o \
		ConduitManager.o \
		Conduit.o \
		InprocConduit.o \
		XprocConduit.o \
		InprocConduit_Write.o \
		InprocConduit_BWrite.o \
		InprocConduit_PWrite.o	\
		InprocConduit_Read.o \
		InprocConduit_Async.o \
		XprocConduit_Async.o \
		Timer.o  \
		Barrier.o \
		SpinBarrier.o \
		SharedDataLock.o \
		SpinLock.o
		
#OBJ_UTC = $(addprefix ./lib/, $(OBJS))

### library required object files
libutc.a: $(OBJ_UTC)
	ar -r -uv libutc.a $(OBJ_UTC)
	cp libutc.a $(PROJECT_LIBDIR)
	cp $(OBJS) $(PROJECT_LIBDIR)
	@echo "make utc library successful !!!" 
ProcList.o: ProcList.cc
	$(C++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
UniqueExeTag.o: UniqueExeTag.cc
	$(C++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
RootTask.o: RootTask.cc
	$(C++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
TaskBase.o: TaskBase.cc
	$(C++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
TaskManager.o: TaskManager.cc
	$(C++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
UtcContext.o:UtcContext.cc
	$(C++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
UtcMpi.o: UtcMpi.cc
	$(C++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)  
Task.o: Task.cc Task.inc
	$(C++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
TaskCPU.o: TaskCPU.cc
	$(C++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
TaskUtilities.o: TaskUtilities.cc
	$(C++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
UserTaskBase.o: UserTaskBase.cc
	$(C++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
ConduitManager.o: ConduitManager.cc
	$(C++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
Conduit.o: Conduit.cc
	$(C++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
InprocConduit.o: InprocConduit.cc
	$(C++)	-o $@ -c $^ $(CCFLAG) $(INCLUDE)
XprocConduit.o: XprocConduit.cc
	$(C++)	-o $@ -c $^ $(CCFLAG) $(INCLUDE)
InprocConduit_Write.o: InprocConduit_Write.cc
	$(C++)	-o $@ -c $^ $(CCFLAG) $(INCLUDE)
InprocConduit_BWrite.o: InprocConduit_BWrite.cc
	$(C++)	-o $@ -c $^ $(CCFLAG) $(INCLUDE)
InprocConduit_PWrite.o: InprocConduit_PWrite.cc
	$(C++)	-o $@ -c $^ $(CCFLAG) $(INCLUDE)
InprocConduit_Read.o: InprocConduit_Read.cc
	$(C++)	-o $@ -c $^ $(CCFLAG) $(INCLUDE)
InprocConduit_Async.o: InprocConduit_Async.cc
	$(C++)	-o $@ -c $^ $(CCFLAG) $(INCLUDE)
XprocConduit_Async.o: XprocConduit_Async.cc
	$(C++)	-o $@ -c $^ $(CCFLAG) $(INCLUDE)	
Timer.o: Timer.cc
	$(C++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
Barrier.o: Barrier.cc
	$(C++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
SpinBarrier.o: SpinBarrier.cc
	$(C++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
SharedDataLock.o: SharedDataLock.cc
	$(C++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
SpinLock.o: SpinLock.cc
	$(C++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)