/*
 *  Author : Ayush Dobhal and Ninad Khargonkar
 *  Date created : 02/25/2020
 *  Description : This file contains implementation of KNN Submodular function.
 */
#include <iostream>
#include <random>
#include <fstream>
#include <string>
#include "parser.h"
#include "../src/optimization/discFunctions/FacilityLocationSparse.h"
#include "../src/representation/Set.h"
#include "../src/representation/SparseGraph.h"
#include "../src/optimization/discAlgorithms/lazyGreedyMax.h"

double get_sqnorm_discrete(std::vector<double> &x1, std::vector<double> &x2) {
    // HAMMING DISTANCE
    double sqnorm = 0;    
    //assert(x1.size() == x2.size());
    for(int i=0; i < x1.size(); ++i){
        if (x1[i] != x2[i]){
            sqnorm += 1;      }
    }
    int N = x1.size();
    if(N > 0){
        sqnorm /= N;
    }
    return sqnorm;
}

double get_sqnorm_continuous(std::vector<double> &x1, std::vector<double> x2) {
    double sqnorm = 0;
    for (int i = 0; i < x1.size(); i++) {
        sqnorm += (x1[i] - x2[i]) * (x1[i] - x2[i]);
    }
    return sqnorm;
}

std::vector<struct datk::SparseGraphItem> getKernel(
        std::vector< std::vector<double> > &dataset1, 
        std::vector<double> &y1, 
        std::vector< std::vector<double> > &dataset2, 
        std::vector<double> &y2,
        int datasetType = 0) {
    
    // datasetType = 0 (discrete), 1 (continuous), or 2 (newsgroup type sparse vec)
    float maxnorm = 0;
    int n = dataset1.size();
    int m = dataset2.size();
    std::vector<datk::SparseGraphItem> kernel(n);
    for(int i=0;  i < n; ++i) {
        kernel[i].index = i;
        kernel[i].num_NN_items = 0;
        // dataset2 will be validtn data U.
        for (int j = 0; j < m; ++ j) {
            if (y1[i] != y2[j]) {
                continue;
            }
            else {  // add to item i's struct only if the labels of i and j match       
                float sqnorm = 0;
                if (datasetType == 0){
                    sqnorm = get_sqnorm_discrete(dataset1[i], dataset2[j]);                
                } else if (datasetType == 1){
                    sqnorm = get_sqnorm_continuous(dataset1[i], dataset2[j]);                
                }
                kernel[i].NNIndex.push_back( j );
                kernel[i].NNSim.push_back(sqnorm);
                kernel[i].num_NN_items += 1;
                if (maxnorm < sqnorm) {
                    maxnorm = sqnorm;
                 }
            }
        } 
    } 
    std::cout << "Max Distance Value: " << maxnorm << "\n";
    // Normalize all distances w(i,j) = maxnorm - w(i,j)
    for (int i = 0; i < n; i++) {        
        for (int j = 0; j < kernel[i].num_NN_items; j++) {
            kernel[i].NNSim[j] = maxnorm - kernel[i].NNSim[j];
        }
    }
    return kernel;
}

 

int main(int argc, char *argv[]) {
    std::string trn_fname (argv[1]);
    std::string val_fname (argv[2]);
    std::string delim (argv[3]);
    double budget = std::stod(argv[4]) ;
    std::string outfile_path (argv[5]);
    int dataType = std::stoi(argv[6]);  // 0 = disc, 1 = cts, 2 = bow 20newsgroups type
    // std::string fname  ("/glusterfs/data/dataset/discrete_datasets/car/trf_car.data.trn");
    // std::string delim (" ");

    std::cout << "Running kNN submod with budget: " << budget << "\n";
    
    Parser trnParse (trn_fname, delim);
    Parser valParse (val_fname, delim);

    std::cout << "Parsed " << budget << "\n";
    
    auto trn_X_data = trnParse.get_X_data();
    auto val_X_data = valParse.get_X_data();
    
    auto trn_Y_data = trnParse.get_Y_data();
    auto val_Y_data = valParse.get_Y_data();
    
    int N_trn = trn_X_data.size();
    int M_trn = trn_X_data[0].size();

    int N_val = val_X_data.size();
    int M_val = val_X_data[0].size();

    std::cout << N_trn << " " << M_trn <<  "\n";
    std::cout << N_val << " " << M_val << "\n";

    std::vector<struct datk::SparseGraphItem> kernel = getKernel(trn_X_data, trn_Y_data, val_X_data, 
            val_Y_data, dataType);
    
    // Note that f_knn is over the validation set. So the "n" here will be its size since that corresponds to
    // the preCompute stats vector which is of length of the outer sum in facility location.   
    datk::FacilityLocationSparse f_knn(N_trn, N_val, kernel);
    std::cout << "Instantiated f_knn \n";

    datk::Set groundSet(N_trn, true);
    datk::Set greedySubset;   // init for the greedySet            
    double subsetSize = N_trn * budget;        
    
    std::cout << "Running lazyGreedy for fraction: " << budget << "\n";    
    datk::lazyGreedyMax(f_knn, subsetSize, greedySubset, 0);
    std::cout << "ArgMax subset computed. Size =  " << greedySubset.size() << "\n";
    
    // std::cout << "Printing subset indices: ";
    // for(auto v: greedySubset){
    //     std::cout << v << " ";
    // }
    // std::cout << "\n";
    
    
    double fNN_groundSet = f_knn.eval(groundSet);    
    std::cout << "Function Value on ground set : " << fNN_groundSet << "\n";

    double fNN_value_greedy = f_knn.eval(greedySubset);
    std::cout << "Function Value on greedy subset: " << fNN_value_greedy << "\n";
    
    // Save the greedy subset. 
    // Convention: Directory = trn data dir. FileName = "file_nb_budget0.8.subset" (if the budget = 0.8)
    //std::string outfile = outfile_path + "file_knn_"+std::to_string((int)(budget*100))+".subset";    
    std::ofstream file;
    std::cout << "filePath " << outfile_path << std::endl;
    file.open(outfile_path);   
    for (auto it = greedySubset.begin(); it != greedySubset.end(); ++it) {
        if (it == greedySubset.begin()){
            file << *it ;   // exclude the comma (delimiter) if its the first element of the set.
        } else {
            file << "," << *it;
        }        
    }
    file.close();

    return 0;
}

