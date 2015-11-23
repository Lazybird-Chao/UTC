#include "RankList.h"
#include "UtcException.h"

namespace iUtc{

    /// \brief Default Constructor
    /// More initialization needs to be done before an object created with
    /// this constructor can be used.
    RankList::RankList()
    {
    }

    // \brief Constructor
    // Construct a RankList, a number of ranks.  A RankList containing
    // 0 will be constructed.
    RankList::RankList(int size)
    {
        for(int i = 0; i < size; i++)
        {
            m_rankList.push_back(0);
        }
    }

    RankList::RankList(int size, int procrank)
	{
		for(int i = 0; i < size; i++)
		{
			m_rankList.push_back(static_cast<RankId_t>(procrank));
		}
	}

    // \brief Constructor
    // Construct a RankList, given a size and C-style array of RankIds.
    RankList::RankList(int size, const RankId_t * ranks)
    {
        RankId_t * tmpPtr = const_cast<RankId_t *>(ranks);
        for(int i = 0; i < size; i++)
        {
            m_rankList.push_back(*(tmpPtr++));
        }
    }

    // \brief Constructor
    // Construct a RankList, given an STL vector of ranks.
    RankList::RankList(const std::vector<RankId_t> & ranks)
    : m_rankList(ranks)
    {
    }

    // \brief Copy constructor
    // Create a RankList, given another RankList.
    RankList::RankList(const RankList & other)
    : m_rankList(other.m_rankList)
    {
    }

    // \brief hasRank
    // Return a boolean indicating if the supplied rank is held in
    // the RankList.
    bool RankList::hasRank(int rank) const
    {
        for(unsigned int i = 0; i < m_rankList.size(); i++)
        {
            if(m_rankList[i] == rank)
            {
                return true;
            }
        }
    return false;
    }

    // \brief setRank
    // Set a rank value in the RankList.
    void RankList::setRank(unsigned int index, unsigned int value)
    {
        if(index < m_rankList.size())
        {
            m_rankList[index] = value;
        }
        else
        {
            throw UtcException("RankList::setRank: Invalid index supplied.",
              __FILE__, __LINE__);
        }
    }

    // \brief getElement
    // Return the rank at the requested position.
    RankId_t RankList::getRank(int index) const
    {
            return m_rankList[index];
    }

    // \brief getNumRanks
    // Return the size of the RankList.
    int RankList::getNumRanks() const
    {
        return m_rankList.size();
    }

    // \brief getRankListVector
    // Return the rank at the requested position.
    void RankList::getRankListVector(std::vector<RankId_t> & rankListCopy) const
    {
           rankListCopy = m_rankList;
    }



    // \brief Assignment operator
    const RankList & RankList::operator=(const RankList & other)
    {
        m_rankList = other.m_rankList;
        return (*this);
    }

    // \brief Equality operator
    bool RankList::operator==(const RankList & other) const
    {
        return (m_rankList == other.m_rankList);
    }

    // \brief Inequality operator
    bool RankList::operator!=(const RankList & other) const
    {
        return (m_rankList != other.m_rankList);
    }

    // \brief Destructor
    RankList::~RankList()
    {
    	m_rankList.clear();
    }
}// namespace iUtc

