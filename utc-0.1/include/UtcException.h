#ifndef UTC_EXCEPTION_H_
#define UTC_EXCEPTION_H_

#include "UtcBasics.h"
#include <iostream>
#include <ostream>
#include <string>
#include <stdexcept>

using std::string;
using std::ostream;
using std::cerr;
using std::endl;

namespace iUtc{

// Exception base class

    class UtcException : public std::exception
    {
    public:
        UtcException(const string& what, const string& file, const int line)
        :m_what(what), m_file(file), m_line(line)
        {
#ifdef PRINT_EXCEPTIONS
            print(cerr);
            cerr << endl;
#endif
        }

        UtcException(const string& what)
        :m_what(what), m_file("Unknown file"), m_line(0)
        {
#ifdef PRINT_EXCEPTIONS
            print(cerr);
            cerr << endl;
#endif
        }

        UtcException(const string& file, const int line)
        :m_what("Unspecified exception"), m_file(file), m_line(0)
        {
#ifdef PRINT_EXCEPTIONS
            print(cerr);
            cerr << endl;
#endif
        }

        UtcException()
        :m_what("Unspecified exception"), m_file("Unknown file"), m_line(0)
        {
#ifdef PRINT_EXCEPTIONS
            print(cerr);
            cerr << endl;
#endif
        }

        virtual ~UtcException() throw(){}

        void print(std::ostream& os) const
        {
            os<<"Exception:"<<m_file<<":"<<m_line<<"  "<<m_what;
        }

        virtual const char* what() const throw()
        {
            return m_what.c_str();
        }

        //friend std::ostream& operator<<(std::ostream& output,
        //    const UtcException& ex);

    private:
        const string m_what;
        const string m_file;
        const int    m_line;


    };// class UtcException

    inline std::ostream& operator<<(std::ostream& output, const iUtc::UtcException& ex)
    {
        ex.print(output);
        return output;
    }



// Specific exception class

    //---------------------------------------------------------------
    // NotImplementedYet exception
    //---------------------------------------------------------------
    class NotImplementedYet : public UtcException
    {
    public:
        NotImplementedYet(const string file, const int line)
            : UtcException("NotImplementedYet",file,line) { }
    };



    //---------------------------------------------------------------
    // BadParameter exception
    //---------------------------------------------------------------
    class BadParameter : public UtcException
    {
    public:
        BadParameter(const string file, const int line)
            : UtcException("BadParameter",file,line) { }
    };

}

#endif


