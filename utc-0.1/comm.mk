##################
PROJECT_HOMEDIR = /home/chao/git/UTC/utc-0.1
PROJECT_BINDIR = $(PROJECT_HOMEDIR)/bin
PROJECT_LIBDIR = $(PROJECT_HOMEDIR)/lib
PROJECT_INCLUDEDIR = $(PROJECT_HOMEDIR)/include
PROJECT_SRCDIR = $(PROJECT_HOMEDIR)/src

##################
GCC = gcc
G++ = g++
MPICXX = mpicxx

CFLAG := -Wall
CCFLAG := -g -std=c++11 -O2

LINKFLAG =

###################
MPI_LIB :=	-pthread -Wl,-rpath -Wl,/opt/openmpi-1.10/lib -Wl,--enable-new-dtags \
				-L/opt/openmpi-1.10/lib -lmpi_cxx -lmpi
BOOST_LIB := /opt/boost-1.60/lib/libboost_thread.a \
				/opt/boost-1.60/lib/libboost_system.a \
				/opt/boost-1.60/lib/libboost_filesystem.a
				
LIB = $(PROJECT_LIBDIR)/libutc.a
LIB += -lrt
LIB += $(MPI_LIB) $(BOOST_LIB)
				
###################		
INCLUDE = -I/opt/openmpi-1.10/include
INCLUDE += -I/opt/boost-1.60/include
INCLUDE += -I$(PROJECT_HOMEDIR)/include		



#####################
.PHONY: cleanlib cleanlog cleanbin

cleanlib:
	rm  $(PROJECT_LIBDIR)/*
cleanlog:
	rm  $(PROJECT_HOMEDIR)/log/*
cleanbin:
	rm  $(PROJECT_BINDIR)/*
		