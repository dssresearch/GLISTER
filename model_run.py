'''
    Author : Ayush Dobhal and Ninad Khargonkar
    Date created : 03/09/2020
    Description : This file contains code to run data selection on small datasets and fit KNN and NB
    model for different budgets and generate plots.
'''
import json
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier


def getFitModel(model_key, model_dict, indices_file, trndata_file, budget, is_random=0):    
    data = np.genfromtxt(trndata_file, delimiter=' ')
    n = data.shape[0]
    if is_random == 0:
        idxs = np.genfromtxt(indices_file, delimiter=',', dtype=int) # since they are indices!
    else:
        idxs = np.random.randint(low=0, high=n-1, size=int(n*budget))
    print("Org data and idxs shape: ", data.shape, " ", idxs.shape)

    if budget != 1:
        data = data[idxs]
        print("subset data shape: ", data.shape)
    else:
        print("subset = full,  data shape: ", data.shape)
        
    X = data[:, : -1]
    Y = data[:, -1]
    model = model_dict[model_key]
    model.fit(X, Y)
    predicted = model.predict(X)
    print("Training Accuracy: " + str(np.mean(predicted==Y)))
    return model
    

def main():
    input_file = "./model_run.input"
    with open(input_file) as f:
        input_dict = json.load(f)
    
    dataset_name = input_dict["dataset_name"] # one value from: car, cifar10, cifar100, mnist, 20ng
    dataset_type = input_dict["dataset_type"] # one value from: discrete, continuous, sparse
    dataset_trn = input_dict["dataset_train"] # string path to trn data
    dataset_val = input_dict["dataset_validation"]
    dataset_tst = input_dict["dataset_test"]
    executable_name = input_dict["executable_name"] # array of selection methods: KNNSubmod, NBSubmod
    budgets = input_dict["budgets"] # array of fractions
    run_path = input_dict["run_path"] # string path to run folder
    model_type = input_dict["model_type"]   # one value out of: KNN, NB, LR, DNN

    #output_path = run_path + executable_name[i] + "_" + dataset_name + "/"
    #indices_file = output_path + executable_name[i] + "_" + str((int)(budgets[j]*100)) + ".subset"

    # handling special case of KNN with disc and cts data. Creating the key for models_dict
    model_key = model_type if model_type != "KNN" else model_type + "_" + dataset_type    
    model_dict = {}
    model_dict["NB"] = MultinomialNB()
    model_dict["LR"] = LogisticRegression(n_jobs=2, solver='sag')
    model_dict["KNN_discrete"] = KNeighborsClassifier(n_neighbors=1, metric='hamming', n_jobs=2)
    model_dict["KNN_continuous"] = KNeighborsClassifier(n_neighbors=1, n_jobs=2)

    print("Running all experiments")
    for i in range(len(executable_name)):
        output_path = run_path + executable_name[i] + "_" + dataset_name + "/"
        log_file = output_path + model_type +"_" + executable_name[i] + "_" + dataset_name + ".log"
        csv_file = output_path + model_type +"_" + executable_name[i] + "_" + dataset_name + ".csv"
        submod_selection_accuracy = []
        random_selection_accuracy = []
        for j in range(len(budgets)):
            indices_file = output_path + executable_name[i] + "_" + str((int)(budgets[j]*100)) + ".subset"
           
            print("Training the model using subset")
            model = getFitModel(model_key, model_dict, indices_file, dataset_trn, budgets[j], is_random=0)

            print("Testset evaluation of model")
            test_dataset = np.genfromtxt(dataset_tst, delimiter=' ')
            X_tst = test_dataset[:, : -1]
            Y_tst = test_dataset[:, -1]
            test_prediction = model.predict(X_tst)
            tst_accuracy = np.mean(test_prediction == Y_tst)
            print("Test accuracy for subset selection = "+str(tst_accuracy))
            print("\n")
            
            if budgets[j] != 1:
                r_model = getFitModel(model_key, model_dict, indices_file, dataset_trn, budgets[j], is_random=1)
                r_test_prediction = r_model.predict(X_tst)
                r_accuracy = np.mean(r_test_prediction == Y_tst)
            else:
                r_accuracy = tst_accuracy
            print("Test accuracy for random selection = "+str(r_accuracy))
            print("\n")

            with open(log_file, "a") as outfile:
                outfile.write("Budget : "+str(budgets[j])+", Subset model test accuracy : "+str(tst_accuracy) + ", Random model test accuracy :"+str(r_accuracy)+"\n")

            with open(csv_file, "a") as cfile:
                cfile.write(str(budgets[j])+","+str(tst_accuracy) + ","+str(r_accuracy)+"\n")

            submod_selection_accuracy.append(tst_accuracy)
            random_selection_accuracy.append(r_accuracy)
        
        F = plt.gcf()
        Size = F.get_size_inches()
        F.set_size_inches(Size[0]*2, Size[1]*2, forward=True)

        plt.plot(budgets, submod_selection_accuracy, '-r', label='Submod Subset Sel', marker='o', markerfacecolor ='g', markeredgecolor ='g', markersize=3)
        plt.plot(budgets, random_selection_accuracy, '-b', label='Random Subset Sel',marker='s', markerfacecolor ='g', markeredgecolor ='g', markersize=3)
        plt.legend(loc="upper left")
        plt.xlabel('Budgets')
        plt.ylabel('Accuracy')
        splitArr = dataset_val.split('\\')
        plt.title(model_type+' model with '+executable_name[i]+' selection on '+dataset_name+' with val='+splitArr[len(splitArr)-1])
        for a,b in zip(budgets, submod_selection_accuracy): 
            plt.text(a, b, str(b), fontsize=8, bbox=dict(facecolor='red', alpha=0.5))
                
        for a,b in zip(budgets, random_selection_accuracy): 
            plt.text(a, b, str(b), fontsize=8, bbox=dict(facecolor='blue', alpha=0.5))
                        
        plt.savefig(model_type+'_'+executable_name[i]+'_'+dataset_name+'.png')
        plt.clf()

if __name__ == "__main__":
    main()
