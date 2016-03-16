#include "UtcMpi.h"
#include "UtcBasics.h"
#include <string>
#include <cstring>
#include <sys/types.h>
#include <unistd.h>

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
            mpi_mode=MPI::THREAD_MULTIPLE;
#else
            mpi_mode=MPI::THREAD_SERIALIZED;
#endif
            int provided=0;
            MPI_Init_thread(&argc, &argv, mpi_mode, &provided);
            //provided=MPI::Init_thread(argc, argv, mpi_mode);
            m_rank= this->rank();
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

        }

        void Utc::finalize()
        {
            if(--nCount !=0)
                return;

            //
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Finalize();

#ifdef USE_DEBUG_LOG
            std::cout<<"[SYSTEM LOG]>>>>>>>>>:";
            std::cout<<"MPI Proc "<<m_rank<<" exit!"<<std::endl;
#endif
        }

        int Utc::rank()
        {
            int rank=0;
            rank = MPI::COMM_WORLD.Get_rank();
            return rank;
        }

        int Utc::rank(MPI::Comm& comm)
        {
            int rank=0;
            rank = comm.Get_rank();
            return rank;
        }

        int Utc::numProcs()
        {
            int procs=0;
            procs = MPI::COMM_WORLD.Get_size();
            return procs;
        }

        int Utc::numProcs(MPI::Comm& comm)
        {
            int procs=0;
            procs = comm.Get_size();
            return procs;
        }

        void Utc::getProcessorName(std::string& name)
        {
            name.clear();
            name.append(m_name);
        }


    }//namespace utcmpi
}//namespace iUtc

