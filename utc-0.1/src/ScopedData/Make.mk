#####################
default:libutc-scopeddata.a

include ../../for-dis.comm

#####################
vpath %.h  $(PROJECT_INCLUDEDIR)
vpath %.inc $(PROJECT_INCLUDEDIR)
vpath %.cc $(PROJECT_SRCDIR)
vpath %.h  $(PROJECT_INCLUDEDIR)/ScopedData
vpath %.inc $(PROJECT_INCLUDEDIR)/ScopedData
vpath %.cc $(PROJECT_SRCDIR)/ScopedData
ifeq ($(ENABLE_INTERNAL_SHMEM), 1)
	vpath %.h  $(PROJECT_INCLUDEDIR)/ScopedData/internal_shmem
	vpath %.inc  $(PROJECT_INCLUDEDIR)/ScopedData/internal_shmem
	vpath %.cc  $(PROJECT_SRCDIR)/ScopedData/internal_shmem
endif
ifeq ($(ENABLE_GPU), 1)
    vpath %.h  $(PROJECT_INCLUDEDIR)/gpu
    vpath %.inc $(PROJECT_INCLUDEDIR)/gpu
    vpath %.cc $(PROJECT_SRCDIR)/gpu
endif


#####################
OBJS:= 	PrivateScopedData.o \
    	GlobalScopedData.o
		
libutc-scopeddata.a: $(OBJS)
	ar -r -uv libutc-scopeddata.a $(OBJS)
	mv libutc-scopeddata.a $(PROJECT_LIBDIR)
	cp $(OBJS) $(PROJECT_LIBDIR)
	@echo "make scopeddata library succefful !!!"
PrivateScopedData.o: PrivateScopedData.cc PrivateScopedData.h PrivateScopedData.inc
	$(C++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
GlobalScopedData.o: GlobalScopedData.cc GlobalScopedData.h GlobalScopedData.inc GlobalScopedData_internal.inc
	$(C++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)

	
clean:
	rm -rf *.o *.a
	
