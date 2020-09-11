import json
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier


def getFitModel(model_key, model_dict, indices_file, trndata_file, budget, is_random=0):    
    data = np.genfromtxt(trndata_file, delimiter=' ')
    n = data.shape[0]
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
    iterations = input_dict["iterations"]

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
    random_selection = numpy.zeroes(iterations)
    run_sum = 0
    for i in range(len(executable_name)):
        output_path = run_path + executable_name[i] + "_" + dataset_name + "/"
        log_file = output_path + model_type +"_" + executable_name[i] + "_" + dataset_name + "_random.log"
        for j in range(len(budgets)):
            for idx in range(iterations):
                print("Testset evaluation of model")
                test_dataset = np.genfromtxt(dataset_tst, delimiter=' ')
                X_tst = test_dataset[:, : -1]
                Y_tst = test_dataset[:, -1]
                r_model = getFitModel(model_key, model_dict, indices_file, dataset_trn, budgets[j], is_random=1)
                r_test_prediction = r_model.predict(X_tst)
                r_accuracy = np.mean(r_test_prediction == Y_tst)
                print("Test accuracy for random selection = "+str(r_accuracy))
                print("\n")
                run_sum += r_accuracy

            with open(log_file, "a") as outfile:
                outfile.write("Budget : "+str(budgets[j])+", Random model test accuracy :"+str(r_accuracy/iterations)+"\n")
                outfile.close()
      
if __name__ == "__main__":
    main()


