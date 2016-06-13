#include "TaskCPU.h"
#include "UtcContext.h"

namespace iUtc{

TaskCPU::TaskCPU(int numLocalThreads,
				 int currentProcessRank,
				 std::vector<ThreadRank_t> tRankList,
				 std::vector<ThreadId_t> *LocalThreadList,
				 std::map<ThreadId_t, ThreadRank_t> *LocalThreadRegistry,
				 std::map<ThreadRank_t, int> *ThreadRank2Local,
				 std::ofstream *procOstream){

	m_numLocalThreads = numLocalThreads;
	m_currentProcessRank = currentProcessRank;
	m_tRankList = tRankList;
	m_LocalThreadList = LocalThreadList;
	m_LocalThreadRegistry = LocalThreadRegistry;
	m_ThreadRank2Local = ThreadRank2Local;
	m_procOstream = procOstream;
}

TaskCPU::~TaskCPU(){
	m_numLocalThreads=0;
	m_currentProcessRank = -1;
	m_tRankList.clear();
	m_LocalThreadList= nullptr;
	m_LocalThreadRegistry= nullptr;
	m_ThreadRank2Local = nullptr;
	m_procOstream = nullptr;

}

int TaskCPU::launchThreads(){
	m_threadSync = new boost::barrier(m_numLocalThreads);
	m_jobDoneWait = new boost::latch(m_numLocalThreads);
	m_jobQueue.clear();
	m_jobHandleQueue.clear();
	m_threadJobIdx = new int[m_numLocalThreads];
	for(int i=0;i<m_numLocalThreads;i++)
		m_threadJobIdx[i]=0;

	////
#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_procOstream)
	(*m_procOstream)<<"start launching threads for ["<<m_Name<<"] on proc "<<m_processRank
			<<"..."<<std::endl;
#endif
	//
	std::unique_lock<std::mutex> LCK(m_activeLocalThreadMutex);
	m_activeLocalThreadCount = m_numLocalThreads;
	LCK.unlock();
	int trank=0;
	ThreadId_t id;
	for(int i=0; i<m_numLocalThreads; i++)
	{
		trank=tRankList[i];
#ifdef USE_DEBUG_LOG
		boost::filesystem::path log_path("./log");
		if(!boost::filesystem::exists(log_path))
			boost::filesystem::create_directory(log_path);
		std::string filename = "./log/";
		filename.append(m_Name);
		filename.append("-");
		filename.append(std::to_string(m_TaskId));
		filename.append("-thread");
		filename.append(std::to_string(trank));
		filename.append(".log");
		std::ofstream *output = new std::ofstream(filename);
		filename.clear();
#else
		std::ofstream *output = nullptr;
#endif

		m_ThreadRank2Local.insert(std::pair<ThreadRank_t, int>(trank, i));
		// crteate a thread
#ifdef SET_CPU_AFFINITY
		int hardcoreId = UtcContext::HARDCORES_ID_FOR_USING;
		UtcContext::HARDCORES_ID_FOR_USING = (UtcContext::HARDCORES_ID_FOR_USING+1)%UtcContext::HARDCORES_TOTAL_CURRENT_NODE;
		m_taskThreads.push_back(std::thread(&Task<T>::threadImpl, this, trank, i, output, hardcoreId));
#else
		m_taskThreads.push_back(std::thread(&Task<T>::threadImpl, this, trank, i, output, -1));
#endif
		//m_taskThreads[i].detach();
		// record some info
		id = m_taskThreads.back().get_id();
		m_LocalThreadList.push_back(id);
		m_LocalThreadRegistry.insert(std::pair<ThreadId_t, ThreadRank_t>(id, trank));
	}

#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_procOstream)
	(*m_procOstream)<<m_numLocalThreads<<" threads for ["<<m_Name<<"] launched on proc "<<m_processRank
			<<std::endl;
#endif
}


}// end namespace iUtc
