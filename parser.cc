#include "parser.h"

Parser :: Parser(std::string path, std::string delimiter, bool include_labels) {
    std::string line;
    include_Y = include_labels;
    filePath = path;
    file.open(filePath, std::ifstream::in);
    while (file) {
        getline(file, line);
        if (line == "-1") break;
        if (!isEmptyString(line)) {
            addInstance(line, delimiter);
        }
    }
    file.close();
}


void Parser :: addInstance(std::string line, std::string delimiter) {
    auto instance = splitString(line, delimiter);
    std::vector<double> x_data;     // should be same type as instance
    int size = instance.size();

    Y_data.push_back(instance[size-1]);
        
    // Also push the Y data at the end -- needed for NB
    // Else just change to: for (int i = 0; i < size-1; i++) {
    int endIdx = size;
    if (include_Y) {
        endIdx = size;
    } else {
        endIdx = size - 1;
    }    
    for (int i = 0; i < endIdx; i++) {
        x_data.push_back(instance[i]);        
    }
    X_data.push_back(x_data);
}


std::vector<double> Parser :: splitString(std::string line, std::string delimiter) {
    size_t pos = 0;
    std::string token;
    std::vector<double> instance; // return value
    while ((pos = line.find(delimiter)) != std::string::npos) {
        token = line.substr(0, pos);
        if (!isEmptyString(token)) {
            instance.push_back(std::stod(token));
        }
        line.erase(0, pos + delimiter.length());
    }
    instance.push_back(std::stod(line));
    return instance;
}


bool Parser :: isEmptyString(std::string str) {
    if(str.find_first_not_of(' ') != std::string::npos) {
        return false;
    }
    return true;
}


std::vector<std::vector<double>> Parser::get_X_data() {
    return X_data;
}


std::vector<double> Parser :: get_Y_data() {
    return Y_data;
}


std::vector<int> Parser::get_X_cardinality() {
    std::vector<int> xcards;
    std::vector< std::unordered_set<double> > X_stats;  // vector of sets of values across columns
    int n = X_data.size();
    int m = X_data[0].size();
    
    for(int j=0; j < m; ++j){
        std::unordered_set<double> colset;
        for(int i=0; i < n; ++i){            
            colset.insert(X_data[i][j]);
        }
        X_stats.push_back(colset);
        xcards.push_back(colset.size());
    }    
    return xcards;
}


int Parser::get_Y_cardinality() {
    std::unordered_set<double> Y_stats;    // set of the Y values (labels), redundant if `include_Y == true`
    int n = Y_data.size();
    for(int i=0; i < n; ++i){
        Y_stats.insert(Y_data[i]);
    }
    return Y_stats.size();
}


void Parser :: printDataset() {
    int n = X_data.size();
    int m = X_data[0].size();
    std::cout << "Size is:" << n << " " << m << std::endl ;

    for (int i = 0; i < m; i++) {
        std::cout << "x_" << i << " ";
    }
    std::cout << "Y\n";
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            std::cout << X_data[i][j] << " ";
        }
        if(! include_Y){
            std::cout << Y_data[i] << std::endl; 
        }    
    }    
}


