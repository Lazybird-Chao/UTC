#####################
default:libutcgpu.a

include ../../make.comm

#####################
vpath %.h  $(PROJECT_INCLUDEDIR)
vpath %.inc $(PROJECT_INCLUDEDIR)
vpath %.cc $(PROJECT_SRCDIR)
vpath %.h  $(PROJECT_INCLUDEDIR)/gpu
vpath %.inc $(PROJECT_INCLUDEDIR)/gpu
vpath %.cc $(PROJECT_SRCDIR)/gpu



#####################
OBJS:= 	CudaDeviceManager.o \
		TaskGPU.o \
		UtcGpuContext.o \
		
libutcgpu.a: $(OBJS)
	ar -r -uv libutcgpu.a $(OBJS)
	mv libutcgpu.a $(PROJECT_LIBDIR)
	mv $(OBJS) $(PROJECT_LIBDIR)
	@echo "make gpulibrary succefful !!!"
CudaDeviceManager.o: CudaDeviceManager.cc
	$(G++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
TaskGPU.o: TaskGPU.cc
	$(G++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
UtcGpuContext.o : UtcGpuContext.cc
	$(G++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
	