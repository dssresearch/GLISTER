/*
C++ implementation for the Naive Bayes model based (submodular) data subset selection
It utilizes `Feature Based function` class to instantiate an object with appropriate weights.  
*/

#include <iostream>
#include <random>
#include <cassert>
#include <fstream>
#include <cmath>

#include "parser.h"
#include "../src/representation/SparseFeature.h"
#include "../src/representation/Set.h"
#include "../src/optimization/discFunctions/FeatureBasedFunctions.h"
#include "../src/optimization/discAlgorithms/lazyGreedyMax.h"


/*
We dont want to convert our feature vector to a 1-hot vector and then convert back again back to 
a sparse representation. Rather we will "simulate" the conversion to 1-hot and then directly find
out the indices in the 1-hot vector which will be non zero from the indexing scheme which we 
would have used to convert to 1-hot in the first place.

For a Naive Bayes model, the feature for a data instance i along the dimension j is (x^({i)}_j, y^{(i)})
Hence the overall length of feature vector will be: |Y| * |X_1| + |Y| * |X_2| + ... + |Y| * |X_d|
So we iterate over each dimension, with the stride in a dimension determined by the y-label: 
y_label * y_card * xcard[i]. We just then add to this stride, the actual value of x^({i)}_j.

After we are done with a dimension, we add |Y| * |X_j| to the total (running sum)
*/
datk::SparseFeature getSparseFromCompact(const std::vector<double> &xvec, double y_label,
                                         const std::vector<int> &xcard, int y_card) {
    datk::SparseFeature s;
    int dim = xvec.size();     // size of our compact vector and in a way, num of nonzero components in onehot vec    
    int totalCard = 0;  // running sum of cardinalities of variables seen so far
    std::vector<int> position(dim, 0);  // positions in the in the 1-hot vector.
    for(int i=0; i < dim; ++i){
        // int posn_i = xvec[i] + y_label * y_card * xcard[i] ;
        int posn_i = xvec[i] + y_label * xcard[i] ;
        position[i] = totalCard + posn_i;
        totalCard += y_card * xcard[i];   
    }
    s.featureIndex = position;    
    s.featureVec = std::vector<double>(dim, 1);   //all features are essentially one or zero
    s.index = 0;    // SETTING IT TO ZERO. SHOULD CORRESPOND TO THE ROW NUMBER!!!
    s.numFeatures = totalCard;
    s.numUniqueFeatures = s.featureVec.size();    
    return s;
}



/* 
Function to get the Sparse Features from a Bow type file -- Bag of Words
Each row in the file corresponds to a document and the format is: 
word_id1 score_id1 word_id2 score_id2 .... word_id_W score_id_W. 
So this is the format that xvec is assumed to have below. We also take in the y-label (class) as info since 
the feature space is over |W| * |Y|. The dense feature vector here will be of length |W| * |Y|. vocab_size = W
*/
datk::SparseFeature getSparseFromBow(const std::vector<double> &xvec, double y_label, 
        int y_card, double vocab_size) {
    datk::SparseFeature s;    
    int n = xvec.size();     
    for(int i=0 ; i < n ; i += 2){        
        double word_id = xvec[i];
        double score_id = xvec[i+1]; 
        if(word_id >= vocab_size){
            continue;
        } else {
            int posInNew = floor(y_card * word_id) + y_label;
            s.featureIndex.push_back(posInNew);
            s.featureVec.push_back(score_id);
        }
    }

    s.index = 0;    // SETTING IT TO ZERO. SHOULD CORRESPOND TO THE ROW NUMBER!!!
    s.numFeatures = y_card * vocab_size;
    s.numUniqueFeatures = s.featureVec.size();    
    return s;
}


std::vector<double> get_featureWeights(const std::vector<struct datk::SparseFeature> &feats){
    int N = feats.size(); // num data points
    int numFeatures = feats[0].numFeatures;
    std::vector<double> feature_weights(numFeatures, 0);
    // Go over the dataset and update the stats for features appearing for a particulat data row (i)
    for(int i=0; i < N; ++i){
        for(int j=0; j < feats[i].numUniqueFeatures; ++j){            
            feature_weights[feats[i].featureIndex[j]] += feats[i].featureVec[j];
        }
    }
    return feature_weights;
}


int main(int argc, char *argv[]) {

    // std::string trn_fname  ("./data/car/trf_car.data.trn");
    // std::string val_fname  ("./data/car/trf_car.data.val");
    // std::string delim (" ");

    // cmd line convention: ./a.out "trnfile_path" "valfile_path" "delimiter" "budget"
    std::string trn_fname (argv[1]);
    std::string val_fname (argv[2]);
    std::string delim (argv[3]);
    double budget = std::stod(argv[4]) ;
    std::string outfile_path = argv[5];
    int dataType = std::stoi(argv[6]);  // 0 = disc, 1 = cts, 2 = bow 20newsgroups type
    std::cout << "Running NB submod with budget: " << budget << "\n";
        
    Parser trnParse (trn_fname, delim);
    Parser valParse (val_fname, delim);
    // std::vector< std::vector<int> > 
    auto trn_X_data = trnParse.get_X_data();
    auto val_X_data = valParse.get_X_data();
    // std::vector<int> 
    auto trn_Y_data = trnParse.get_Y_data();
    auto val_Y_data = valParse.get_Y_data();
    // std::vector<int> 
    auto X_card = trnParse.get_X_cardinality();
    int Y_card = trnParse.get_Y_cardinality();

    // These stats applicable when dataType != 2.
    int N_trn = trn_X_data.size();
    int M_trn = trn_X_data[0].size();
    int N_val = val_X_data.size();
    int M_val = val_X_data[0].size();

    // Only appicable when dataType = 2
    double vocab = 0;
    for(int i =0; i < N_trn; ++i){
        for(int j=0; j < trn_X_data[i].size(); j+=2){
            double word_id = trn_X_data[i][j];
            if(vocab < word_id){
                vocab = word_id;
            }
        }
    }

    std::vector<struct datk::SparseFeature> trn_sparse_feats (N_trn);
    for(int i=0; i < N_trn; ++i){
        datk::SparseFeature s_feat;
        if (dataType == 2){
            s_feat = getSparseFromBow(trn_X_data[i], trn_Y_data[i], Y_card, vocab);
        }        
        else {
            s_feat = getSparseFromCompact(val_X_data[i], val_Y_data[i], X_card, Y_card);
        }
        trn_sparse_feats[i] = s_feat;  
    }

    std::vector<struct datk::SparseFeature> val_sparse_feats (N_val);
    for(int i=0; i < N_val; ++i){
        datk::SparseFeature s_feat;
        if(dataType == 2){
            s_feat = getSparseFromBow(val_X_data[i], val_Y_data[i], Y_card, vocab);
        } else {
            s_feat = getSparseFromCompact(val_X_data[i], val_Y_data[i], X_card, Y_card);
        }        
        val_sparse_feats[i] = s_feat;  
    }



    std::cout << "Collected the sparse features" << std::endl;     
    std::vector<double> val_feat_weights = get_featureWeights(val_sparse_feats);    
    std::cout << "Computed feature weights" << std::endl; 
    
    datk::FeatureBasedFunctions f_NBval (N_trn, 3, trn_sparse_feats, val_feat_weights); // FNB using U, Just the weights change!
    std::cout << "Instantiated F_NB" << std::endl; 
      
    // Not really needed but useful to check for upper bound on function value
    datk::Set groundSet(N_trn, true);
    auto fNB_groundSet = f_NBval.eval(groundSet);   
    std::cout << "Function Value on ground set : " << fNB_groundSet << "\n\n";

    
    std::cout << "Running lazyGreedy for fraction: " << budget << "\n";
    datk::Set greedySubset = datk::Set();   // init for the greedySet            
    double subsetSize = N_trn * budget;        
    datk::lazyGreedyMax(f_NBval, subsetSize, greedySubset, 0);        
    std::cout << "ArgMax subset computed. Size =  " << greedySubset.size() << "\n";
    auto fNB_value_greedy = f_NBval.eval(greedySubset);       
    std::cout << "Function Value on greedy subset: " << fNB_value_greedy << "\n";        

    // Save the greedy subset. 
    // Convention: Directory = trn data dir. FileName = "file_nb_budget0.8.subset" (if the budget = 0.8)
    //std::string outfile = output_path+"file_nb_" + std::to_string((int)(budget * 100)) + ".subset" ;
    std::ofstream file;
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
