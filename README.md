# TODO

- **IMP:** Standardizing dataset format convention. Always using csv files with last col
  being the y-label? -- yes?
- Work with the special case of 20newsgroups data: counts already computed! 
- Collect results for different experiments and plot out (python) + short report.
- Code for one-step gradient algorithm -- it assumes a general loss function: Working on
  logisitic loss (binary classification) for now. Can extend it to softmax. 
- Code for DNN will have to written in python (torch) with naive greeedy. 
- Random sampling with equal distribution of each class.

## Final Executable convention

We plan to have the following convention for running the excutables (`NBsubmod` and 
`KNNsubmod`) as follows:

`./KNNsubmod "trn file path" "val file path "delimiter" "budget"`

Optionally we can also include the name of output indices files (with the convention
as : "training-data-dir/model-budget.subset" )

`./KNNsubmod "trn file path" "val file path "delimiter" "budget" "subset-file"`


## Integrating different parts

So, finally we should only have 2 fileers: `nb_submod.cc` and
`knn_submod.cc`. Other specific executables for data selection can be added later on.
These should take in a trn, val dataset (plus parsing arguments)
along with details like: subset size (budget) and whether data is continuous or
discrete data?. If the data is continuous, then we first bin it appropriately 
(only for naive bayes or other discrete selection models) and then run the selection 
algorithm. 

A special case to handle is the 20newsgroups data which is in a somewhat sparse
format in itself i.e it already has the counts! So we can handle this case
separately instead.

After the selection code is run, we need to get accuracy / loss results too. 
So at the end of selection algorithm is run, we save the results (i.e the indices 
of points selected, log-llk value etc) in a json format in an appropriate directory.
We can then load this json file up (from python/C++) to learn the model using just
the best subset (indicated by the indices) and get accuracy/loss results in a 
nice format. There is already sample code in c++ about how to obtain accuracy
results too.

For that after obtaining the index of the data points, write out the results
to a text file. Then from python, load the training data and the indices
from the text file, use only those indices and compute the accuracy results.

So the pipeline will be as follows: 

- Run the data selection algorithm (NB, kNN etc.) wrt validation data and a given 
  budget (take it as a command line argument).
- Write out the results: data points (indices) selected + other info
- Train a model (in Python?) using the above indices -- get accuracy results
- Compile the results across different values for budgets, eg: 0.1, 0.25, 0.5 etc.



