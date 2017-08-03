#include "UtcMpi.h"
#include "UtcBasics.h"
#include <string>
#include <cstring>
#include <sys/types.h>
#include <unistd.h>
#include <iostream>

namespace iUtc{
    namespace UtcMpi{

        int Utc::nCount=0;
        char Utc::m_name[MPI_MAX_PROCESSOR_NAME];

        Utc::Utc()
        {
            int argc =1;
            char* argv_storage[2];
            char** argv = argv_storage;
            argv[0] = (char*)"utcmpi";
            argv[1]= NULL;
            initialize(argc, argv);
        }

        Utc::Utc(int& argc, char**& argv)
        {
            initialize(argc, argv);
        }

        Utc::~Utc()
        {
            finalize();
        }

        void Utc::initialize(int& argc, char** argv)
        {
            if(nCount++ !=0)
                return;

            int mpi_mode;
#ifdef  MULTIPLE_THREAD_MPI
            mpi_mode=MPI_THREAD_MULTIPLE;
#else
            mpi_mode=MPI_THREAD_SERIALIZED;
#endif
            int provided=0;
            MPI_Init_thread(&argc, &argv, mpi_mode, &provided);
            m_rank= this->rank();
            if(m_rank==0){
            	std::cout<< "MPI ThreadMode="<<m_mode[provided]<<std::endl;
            }
            m_size = this->numProcs();
            int length;
            MPI_Get_processor_name(m_name, &length);
            m_name[length]='\0';
#ifdef USE_DEBUG_LOG
            std::cout<<"[SYSTEM LOG]>>>>>>>>>:";
            std::cout<< "MPI ThreadMode="<<m_mode[provided]<<", "
                    <<"Total processes="<<m_size<<", "
                    <<"Current proc rank="<<m_rank<<"("<<getpid()<<")"<<std::endl;
#endif


#if ENABLE_SCOPED_DATA
#ifdef USE_OPENSHMEM
            // for openmpi-1.8.3, call this with mpi_init_thread() before will cause dead
            // lock(bug maybe); shmem_init inside will check if mpi_init called, if not it
            // will call it, further inside it only call mpi_init(), no consider thread
            // support, so we need call mpi_init_thread explicitely in advance, and then
            // call shmem_init
            shmem_init();
#endif
#endif

        }

        void Utc::finalize()
        {
            if(--nCount !=0)
                return;

            //
            MPI_Barrier(MPI_COMM_WORLD);

#if ENABLE_SCOPED_DATA
#ifdef USE_OPENSHMEM
            //shmem_finalize should be called first, and inside will finalize mpi
            // so after this can't call mpi_finalize again
            //std::cout<<ERROR_LINE<<std::endl;
            shmem_finalize();
#endif
#else

            MPI_Finalize();
#endif

#ifdef USE_DEBUG_LOG
            std::cout<<"[SYSTEM LOG]>>>>>>>>>:";
            std::cout<<"MPI Proc "<<m_rank<<" exit!"<<std::endl;
#endif
        }

        int Utc::rank()
        {
            int rank=0;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            return rank;
        }

        int Utc::rank(MPI_Comm& comm)
        {
            int rank=0;
            MPI_Comm_rank(comm, &rank);
            return rank;
        }

        int Utc::numProcs()
        {
            int procs=0;
            MPI_Comm_size(MPI_COMM_WORLD, &procs);
            return procs;
        }

        int Utc::numProcs(MPI_Comm& comm)
        {
            int procs=0;
            MPI_Comm_size(comm, &procs);
            return procs;
        }

        void Utc::getProcessorName(std::string& name)
        {
            name.clear();
            name.append(m_name);
        }

        void Utc::Barrier()
        {
        	MPI_Barrier(MPI_COMM_WORLD);
        }

        void Utc::Barrier(MPI_Comm &comm)
        {
        	MPI_Barrier(comm);
        }


    }//namespace utcmpi
}//namespace iUtc

