/*
 *  Author : Ayush Dobhal and Ninad Khargonkar
 *  Date created : 02/20/2020
 *  Description : This file is the header file for C++ parser for csv files. The .cc file contains
    the required functions needed to parse data from a dataset in csv format.
 */

#ifndef _PARSER_H_
#define _PARSER_H_

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_set>

class Parser {
    private:
        std::string filePath;
        std::ifstream file;
        bool include_Y;
        std::vector<std::vector<double>> X_data;
        std::vector<double> Y_data;    
    
    public :
        Parser(std::string fileName, std::string delimiter, bool include_labels=false);
        void addInstance(std::string line, std::string delimiter);
        bool isEmptyString(std::string);
        void printDataset();
        std::vector<double> splitString(std::string line, std::string delimiter);
        std::vector<std::vector<double>> get_X_data();
        std::vector<double> get_Y_data();
        std::vector<int> get_X_cardinality();
        int get_Y_cardinality();
};

#endif  // _PARSER_H_
