#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <algorithm>
#include <fstream>

#include "parser.h"


std::vector<double> binning(std::vector<double> & in, int N){
    if (N ==1)
        // return in; Incorrect. put everything into a single bin i.e zero! -- meaningless.
        return std::vector<double> (in.size(), 0);

    auto max = *(std :: max_element(in.begin(), in.end())) ;
    auto min = *(std :: min_element(in.begin(), in.end())) ; 
    int width = ceil(fabs(max -min)/N);
    int left =0, right=N;
    std::vector<double> out;

    //std::cout<<max<<" "<<min<<std::endl;

    for (int i=0;i<in.size();i++){
        int left =0; right=N;
        while(left < right){
            //std::cout<<in[i]<<" ";//<<std::endl;
           int  j = (right-left)/2;
            if (in[i] <= min + (left+j+1)*width && in[i] >= min + (left+j)*width){
                out.push_back(left+j);
                break;
            }
            else if(in[i] < min + (left+j)*width){
                right = j;
            }
            else{
                left += j;
            }
        }
    }
    return out;
}

std::vector<double> binning_fast(std::vector<double> & in, int num_bins){
    if (num_bins ==1)
        // return in; Incorrect. put everything into a single bin i.e zero! -- meaningless.
        return std::vector<double> (in.size(), 0);

    auto maxVal = *(std :: max_element(in.begin(), in.end())) ;
    auto minVal = *(std :: min_element(in.begin(), in.end())) ; 
    int width = ceil(fabs(maxVal - minVal)/num_bins);    
    std::vector<double> out(in.size(), 0);    
    for (int i=0;i<in.size();i++){
        int bin_num = floor((in[i] - minVal)/width);
        out[i] = bin_num;
        }
    return out;
}


int main(int argc, char *argv[]) {    

    //execute as ./a.out "input_file_name" "delimiter" "output_file_name" "column_no_1" "no_of_bins_1" ... "column_no_n" "no_of_bins_n"

    std::string fname  (argv[1]); // input_file
    std::string delim (argv[2]);  // delimiter
    std::string fname_out  (argv[3]); // output_file (binned)

    Parser trn (fname, delim);
    std::vector< std::vector<double> > X_data = trn.get_X_data();    
    std::vector<double> Y_data = trn.get_Y_data();
    int N = X_data.size();  // num rows
    int M = X_data[0].size();   // num columns
    std::cout << "Data parsed. X_data Dims: " << N << "," << M << "\n";

    std::vector<int> no_of_bins; // want to handle binning for specific columns    
    std::vector<int> column_no;
    int num_bins = 10;  // default value

    if(argc == 4){ // i.e perform binning on all columns with num_bins = 10 (default value)
        for(int i=0; i < M; ++i){
            column_no.push_back(i);
            no_of_bins.push_back(num_bins);
        }
    } else if (argc == 5) { // perform binning on all columns with num_bins = argc[4] -- non-default value
        num_bins = std::stoi(argv[4]); // change from default val
        for(int i=0; i < M; ++i){
            column_no.push_back(i);
            no_of_bins.push_back(num_bins);
        }
    } else { // variable number of args -- specific columns and bin size passed.
        for (int i=4; i<argc; i+=2){
            column_no.push_back(std::stod(argv[i]));
            no_of_bins.push_back(std::stod(argv[i+1]));
        }
    }

    std::cout<< "Columns and Bins read in" << std::endl;

    for (int i=0; i<no_of_bins.size(); i++){
        std::vector<double> temp;
        for (int j=0; j<N; j++){
            temp.push_back(X_data[j][column_no[i]]);
        }
        // temp = binning(temp, no_of_bins[i]);
        std::vector<double> binned = binning_fast(temp, no_of_bins[i]);
        for (int j=0;j<N;j++){
           X_data[j][column_no[i]] = binned[j] ;
        }
    }

    std::ofstream file;
    file.open(fname_out);
    std::cout << "Binning done. Writing output file: " << fname_out << std::endl;  
    for (int i=0; i<N; i++)
        for (int j=0; j<M+1; j++){  // j goes over 0...M -- since we also want to write out the Y-labels in last col.
            if (j == M)
                file << Y_data[i] << "\n"; // last column == Y label
            else
                file << X_data[i][j] << delim;  
        }    
    file.close();
     return 0;
}