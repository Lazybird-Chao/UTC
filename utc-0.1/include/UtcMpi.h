#ifndef UTC_MPI_H_
#define UTC_MPI_H_

#include <mpi.h>
#include "UtcBase.h"
#include <string>

namespace iUtc{

    namespace UtcMpi{

        class Utc:public iUtc::UtcBase
        {

        public:
            Utc();

            Utc(int& argc, char**& argv);

            ~Utc();

            int numProcs();

            int numProcs(MPI::Comm &comm);

            int rank();

            int rank(MPI::Comm &comm);

            void getProcessorName(std::string& name);

            void Barrier();

            void Barrier(MPI_Comm &comm);

        private:
            Utc(const Utc&);
            Utc& operator=(const Utc&);

            void initialize(int &argc, char** argv);

            void finalize();

            static int nCount;  //no actual meaning now, may be deleted later
            int m_rank;
            int m_size;
            static char m_name[MPI_MAX_PROCESSOR_NAME];
            std::string m_mode[4] = {"MPI_THREAD_SINGLE",
                                "MPI_THREAD_FUNNELED",
                                "MPI_THREAD_SERIALIZED",
                                "MPI_THREAD_MULTIPLE"};

        };
    }//namespace utcmpi
}//namespace iUtc




#endif
