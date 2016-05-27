#include "../include/ProcList.h"
#include "UtcException.h"
#include <iostream>

namespace iUtc{

    /// \brief Default Constructor
    /// More initialization needs to be done before an object created with
    /// this constructor can be used.
	ProcList::ProcList()
    {
		m_procList.clear();
    }

	//
	// Construct a ProcList that only has one member which is proc
	ProcList::ProcList(ProcRank_t proc)
    {
		m_procList.clear();
        m_procList.push_back(proc);

    }

	// Construct a ProcList that contains size copys of proc
	ProcList::ProcList(int size, ProcRank_t proc)
	{
		if(size <1){
			std::cerr<<"ERROR: number of threads must > 0!"<<std::endl;
			exit(1);
		}
		m_procList.clear();
		for(int i = 0; i < size; i++)
		{
			m_procList.push_back(proc);
		}
	}

    // \brief Constructor
    // Construct a RankList, given a size and C-style array of RankIds.
	ProcList::ProcList(int size, const int * ranks)
    {
		m_procList.clear();
		ProcRank_t * tmpPtr = const_cast<ProcRank_t *>(ranks);
        for(int i = 0; i < size; i++)
        {
            m_procList.push_back(*(tmpPtr++));
        }
    }

	void ProcList::push_back(int rank){
		m_procList.push_back(rank);
	}


    // \brief Constructor
    // Construct a RankList, given an STL vector of ranks.
	ProcList::ProcList(const std::vector<ProcRank_t> & ranks)
    : m_procList(ranks)
    {
    }

    // \brief Copy constructor
    // Create a RankList, given another RankList.
	ProcList::ProcList(const ProcList & other)
    : m_procList(other.m_procList)
    {
    }

    // \brief hasRank
    // Return a boolean indicating if the supplied proc is held in
    // the ProcList.
    bool ProcList::hasProc(int proc) const
    {
        for(unsigned int i = 0; i < m_procList.size(); i++)
        {
            if(m_procList[i] == proc)
            {
                return true;
            }
        }
    return false;
    }

    // \brief setRank
    // Set a rank value in the RankList.
    void ProcList::setProc(unsigned int index, unsigned int value)
    {
        if(index < m_procList.size())
        {
            m_procList[index] = value;
        }
        else
        {
            throw UtcException("RankList::setRank: Invalid index supplied.",
              __FILE__, __LINE__);
        }
    }

    // \brief getElement
    // Return the rank at the requested position.
    ProcRank_t ProcList::getProc(int index) const
    {
            return m_procList[index];
    }

    // \brief getNumRanks
    // Return the size of the RankList.
    int ProcList::getNumProcs() const
    {
        return m_procList.size();
    }

    // \brief getRankListVector
    // Return the rank at the requested position.
    void ProcList::getProcListVector(std::vector<ProcRank_t> & rankListCopy) const
    {
           rankListCopy = m_procList;
    }



    // \brief Assignment operator
    const ProcList & ProcList::operator=(const ProcList & other)
    {
        m_procList = other.m_procList;
        return (*this);
    }

    // \brief Equality operator
    bool ProcList::operator==(const ProcList & other) const
    {
        return (m_procList == other.m_procList);
    }

    // \brief Inequality operator
    bool ProcList::operator!=(const ProcList & other) const
    {
        return (m_procList != other.m_procList);
    }

    // \brief Destructor
    ProcList::~ProcList()
    {
    	m_procList.clear();
    }
}// namespace iUtc

