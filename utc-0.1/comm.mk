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
MPI_DIR = /opt/openmpi-1.10
BOOST_DIR = /opt/boost-1.60
OSH_DIR = /opt/openmpi-1.10

MPI_LIB :=	-pthread -Wl,-rpath -Wl,$(MPI_DIR)/lib -Wl,--enable-new-dtags \
				-L$(MPI_DIR)/lib -lmpi_cxx -lmpi -loshmem
BOOST_LIB := $(BOOST_DIR)/lib/libboost_thread.a \
				$(BOOST_DIR)/lib/libboost_system.a \
				$(BOOST_DIR)/lib/libboost_filesystem.a
OSH_LIB :=
				
LIB = $(PROJECT_LIBDIR)/libutc.a
LIB += -lrt
LIB += $(MPI_LIB) $(BOOST_LIB)
				
###################		
INCLUDE = -I$(MPI_DIR)/include
INCLUDE += -I$(BOOST_DIR)/include
INCLUDE += -I$(PROJECT_HOMEDIR)/include		



#####################
.PHONY: cleanlib cleanlog cleanbin

cleanlib:
	rm  $(PROJECT_LIBDIR)/*
cleanlog:
	rm  $(PROJECT_HOMEDIR)/log/*
cleanbin:
	rm  $(PROJECT_BINDIR)/*
		