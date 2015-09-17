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

        //
        static TaskManager* getTaskManager();


    protected:

    private:

        static void initialize(int &argc, char**argv);

        static void finalize();

        static UtcBase* Utcbase_provider;

        static TaskId m_rootTaskId;
        static RootTask* root;

        static int m_nCount;    // may not be useful!

    };// class UtcContext
}//namespace iUtc



#endif /* UTC_UTCCONTEXT_H_ */
