#####################
default:libutcgpu.a

include ../../for-dis.comm

#####################
vpath %.h  $(PROJECT_INCLUDEDIR)
vpath %.inc $(PROJECT_INCLUDEDIR)
vpath %.cc $(PROJECT_SRCDIR)
ifeq ($(ENABLE_SCOPED_DATA), 1)
    vpath %.h  $(PROJECT_INCLUDEDIR)/ScopedData
    vpath %.inc $(PROJECT_INCLUDEDIR)/ScopedData
    vpath %.cc $(PROJECT_SRCDIR)/ScopedData
endif
vpath %.h  $(PROJECT_INCLUDEDIR)/gpu
vpath %.inc $(PROJECT_INCLUDEDIR)/gpu
vpath %.cc $(PROJECT_SRCDIR)/gpu



#####################
OBJS:= 	CudaDeviceManager.o \
		TaskGPU.o \
		UtcGpuContext.o \
		GpuTaskUtilities.o \
		GpuKernel.o
		
libutcgpu.a: $(OBJS)
	ar -r -uv libutcgpu.a $(OBJS)
	mv libutcgpu.a $(PROJECT_LIBDIR)
	cp $(OBJS) $(PROJECT_LIBDIR)
	@echo "make gpulibrary succefful !!!"
CudaDeviceManager.o: CudaDeviceManager.cc CudaDeviceManager.h  UtcGpuBasics.h
	$(C++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
TaskGPU.o: TaskGPU.cc
	$(C++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
UtcGpuContext.o : UtcGpuContext.cc
	$(C++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
GpuTaskUtilities.o : GpuTaskUtilities.cc
	$(C++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
GpuKernel.o : GpuKernel.cc
	$(C++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)

	
clean:
	rm -rf *.o *.a
