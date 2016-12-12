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
	cp libutc-scopeddata.a $(PROJECT_LIBDIR)
	cp $(OBJS) $(PROJECT_LIBDIR)
	@echo "make scopeddata library succefful !!!"
PrivateScopedData.o: PrivateScopedData.cc PrivateScopedData.inc
	$(C++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
GlobalScopedData.o: GlobalScopedData.cc GlobalScopedData.inc
	$(C++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)

	
clean:
	rm -rf *.o *.a
	
