#ifndef UTC_RANK_LIST_H_
#define UTC_RANK_LIST_H_

#include "UtcBasics.h"
#include <vector>

namespace iUtc{

    class RankList
    {

    public:

        RankList();

        /// \brief Constructor
        /// Construct a ThreadRankList, given an STL vector of ranks.
        RankList(const std::vector<RankId_t> &);

        /// \brief Constructor
        /// Construct a RankList, a number of ranks.  A ThreadRankList containing
        /// 0  will be constructed.
        RankList(int size);
        RankList(int size, int proc);

        /// \brief Constructor
        /// Construct a ThreadRankList, given a size and C-style array of RankIds.
        RankList(int size, const RankId_t * ranks);

        /// \brief Copy constructor
        /// Create a RankList, given another RankList.
        RankList(const RankList &);

        /// \brief Destructor
        ~RankList();

        /// \brief getElement
        /// Return the rank at the requested position.
        RankId_t getRank(int index) const;

        /// \brief getNumRanks
        /// Return the size of the RankList.
        int getNumRanks() const;

        /// \brief getRankListVector
        /// Return the rank at the requested position.
        void getRankListVector(std::vector<RankId_t> &) const;

        /// \brief hasRank
        /// Return a boolean indicating if the supplied rank is held in
        /// the RankList.
        bool hasRank(int rank) const;

        /// \brief setRank
        /// Set a rank value in the RankList.
        void setRank(unsigned int index, unsigned int value);


        /// \brief Assignment operator
        const RankList & operator=(const RankList & other);

        /// \brief Equality operator
        bool operator==(const RankList & other) const;

        /// \brief Inequality operator
        bool operator!=(const RankList & other) const;

    private:

        /// \brief m_ranks
        /// Container for the ranks.
        std::vector<RankId_t> m_rankList;



    };// class RankList
}// namespace iUtc


















#endif







