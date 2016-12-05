#####################
default:libutc_scopeddata.a

include ../../make.comm

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
		
libutcgpu.a: $(OBJS)
	ar -r -uv libutc_scopeddata.a $(OBJS)
	cp libutc_scopeddata.a $(PROJECT_LIBDIR)
	cp $(OBJS) $(PROJECT_LIBDIR)
	@echo "make scopeddata library succefful !!!"
PrivateScopedData.o: PrivateScopedData.cc PrivateScopedData.inc
	$(G++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
GlobalScopedData.o: GlobalScopedData.cc GlobalScopedData.inc
	$(G++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)

	
clean:
	rm -rf *.o *.a
	