#ifndef UTC_UTCCONTEXT_H_
#define UTC_UTCCONTEXT_H_

#include "UtcBasics.h"
#include "UtcBase.h"
#include "TaskManager.h"
#include "UtcMpi.h"
#include "RootTask.h"
#include "SharedDataLock.h"
#include "SpinLock.h"
#include "FastMutex.h"

#include <vector>
#include <string>

namespace iUtc{
class UtcContext{
/*TODO:
	* As this class is designed as singleton, no need to use so many
	* static data member and method members.
	* Should change these static to normal members.
	*/
    public:

        ~UtcContext();

        static UtcContext& getContext();
        static UtcContext& getContext(int &argc, char** &argv);

        /*
         * TODO: add this utc overall finish method for explicitly use in main program,
         * 		 do some necessary clean work before program exit.
         * 		 Because put some clean work in destructor with automatic invoke when program
         * 		 is exiting may call system error. (like call cudaSetDevice())
         */
        static void Finish();

        int getProcRank();

        int numProcs();

        void getProcessorName(std::string& name);

        void Barrier();

        //std::mutex* getCtxMutex();
        FastMutex* getCtxMutex();
        SpinLock* getCtxSpinMutex();

        //
        static TaskManager* getTaskManager();

        //
        static int HARDCORES_TOTAL_CURRENT_NODE;
        static int HARDCORES_ID_FOR_USING;

        //
        int getNumGPUs();


    protected:

    private:
        class dummyContext{
        public:
        	int dummy;

        	~dummyContext(){
        		if(m_ContextInstance)
        			delete m_ContextInstance;
        		m_ContextInstance = nullptr;
        	}
        };
        static dummyContext m_dummyInstance;

        UtcContext();

        UtcContext(int &argc, char** &argv);

        static void initialize(int &argc, char**argv);

        static void finalize();

        static UtcBase* Utcbase_provider;

        static TaskId_t m_rootTaskId;
        static RootTask* root;

        static int m_nCount;    // may not be useful!
        static UtcContext *m_ContextInstance;

        //std::mutex m_ctxMutex;
        FastMutex m_ctxMutex;
        SpinLock m_ctxSpinMutex;

        //
        static int m_numGPUs;

        //
        UtcContext(const UtcContext& other);
        UtcContext& operator=(const UtcContext& other);

    };// class UtcContext
}//namespace iUtc



#endif /* UTC_UTCCONTEXT_H_ */
