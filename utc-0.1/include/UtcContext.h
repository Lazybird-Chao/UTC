#ifndef UTC_UTCCONTEXT_H_
#define UTC_UTCCONTEXT_H_

#include "UtcBasics.h"
#include "UtcBase.h"
#include "TaskManager.h"
#include "UtcMpi.h"
#include "RootTask.h"

#include <vector>
#include <string>

namespace iUtc{
class UtcContext{

    public:
        UtcContext();

        UtcContext(int &argc, char** &argv);

        ~UtcContext();

        int getProcRank();

        int numProcs();

        void getProcessorName(std::string& name);

        void Barrier();

        //
        static TaskManager* getTaskManager();

        //
        static int HARDCORES_TOTAL_CURRENT_NODE;
        static int HARDCORES_ID_FOR_USING;
    protected:

    private:

        static void initialize(int &argc, char**argv);

        static void finalize();

        static UtcBase* Utcbase_provider;

        static TaskId_t m_rootTaskId;
        static RootTask* root;

        static int m_nCount;    // may not be useful!

        //
        UtcContext(const UtcContext& other);
        UtcContext& operator=(const UtcContext& other);

    };// class UtcContext
}//namespace iUtc



#endif /* UTC_UTCCONTEXT_H_ */
