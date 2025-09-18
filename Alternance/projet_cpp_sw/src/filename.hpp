#ifndef FILENAME_HPP
#define FILENAME_HPP

#include <string>
#include <sstream>
#include <iomanip>

inline std::string format_filename(const std::string& base, int iter,const std::string& ext ) {
    std::ostringstream oss;
    oss << base << "_" << std::setfill('0') << std::setw(9) << iter << ext;
    return oss.str();
}

#endif 
