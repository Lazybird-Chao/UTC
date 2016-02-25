#ifndef UTC_RANK_LIST_H_
#define UTC_RANK_LIST_H_

#include "UtcBasics.h"
#include <vector>

namespace iUtc{

    class ProcList
    {

    public:

    	ProcList();

        /// \brief Constructor
        /// Construct a ThreadRankList, given an STL vector of ranks.
    	ProcList(const std::vector<ProcRank_t> &);

        /// \brief Constructor
        /// Construct a ProcList that only has one member which is proc
    	ProcList(ProcRank_t proc);
    	/// // Construct a ProcList that contains size copys of proc
    	ProcList(int size, ProcRank_t proc);

        /// \brief Constructor
        /// Construct a ThreadRankList, given a size and C-style array of RankIds.
    	ProcList(int size, const int * ranks);

        /// \brief Copy constructor
        /// Create a RankList, given another RankList.
    	ProcList(const ProcList &);

        /// \brief Destructor
        ~ProcList();

        /// \brief getElement
        /// Return the rank at the requested position.
        ProcRank_t getProc(int index) const;

        /// \brief getNumRanks
        /// Return the size of the RankList.
        int getNumProcs() const;

        /// \brief getRankListVector
        /// Return the rank at the requested position.
        void getProcListVector(std::vector<ProcRank_t> &) const;

        /// \brief hasRank
        /// Return a boolean indicating if the supplied rank is held in
        /// the RankList.
        bool hasProc(int proc) const;

        /// \brief setRank
        /// Set a rank value in the RankList.
        void setProc(unsigned int index, unsigned int value);


        /// \brief Assignment operator
        const ProcList & operator=(const ProcList & other);

        /// \brief Equality operator
        bool operator==(const ProcList & other) const;

        /// \brief Inequality operator
        bool operator!=(const ProcList & other) const;

    private:

        /// \brief m_ranks
        /// Container for the ranks.
        std::vector<ProcRank_t> m_procList;



    };// class RankList
}// namespace iUtc


















#endif







