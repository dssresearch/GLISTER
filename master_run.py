'''
    Author : Ayush Dobhal
    Date created : 4/28/2020
    Description : This file contains code for running neural nets runs for combinations of 
    different datasets, learning rates and budgets. 
'''

import json
import os
import subprocess

#lr = [2, 1, 0.5, 0.25, 0.05, 0.01]
lr = [0.1]
input_file = "./neural_net_run.input"   # Config File for the experiment
with open(input_file) as f:
    input_dict = json.load(f)

datasets = ["mnist"]
num_classes = [10]
indices_file = [
    #"/glusterfs/data/data_subset_selection/subset_data/subset_with_train/NBSubmod_cifar100_bin/NBSubmod_10.subset"
    #"/glusterfs/data/data_subset_selection/subset_data/subset_with_train/NBSubmod_cifar100_bin/NBSubmod_25.subset"
    #"/glusterfs/data/data_subset_selection/subset_data/subset_with_train/NBSubmod_cifar10_bin/NBSubmod_25.subset"
    #"/glusterfs/data/data_subset_selection/subset_data/subset_with_train/NBSubmod_cifar10_bin/NBSubmod_10.subset"
    #"/glusterfs/data/data_subset_selection/subset_data/subset_with_train/KNNSubmod_mnist_cont/KNNSubmod_10.subset",
    #"/glusterfs/data/data_subset_selection/subset_data/subset_with_train/KNNSubmod_mnist_cont/KNNSubmod_15.subset",
    #"/glusterfs/data/data_subset_selection/subset_data/subset_with_train/KNNSubmod_mnist_cont/KNNSubmod_20.subset",
    #"/glusterfs/data/data_subset_selection/subset_data/subset_with_train/KNNSubmod_mnist_cont/KNNSubmod_25.subset",
    #"/glusterfs/data/data_subset_selection/subset_data/subset_with_train/KNNSubmod_mnist_cont/KNNSubmod_30.subset"
    #"/glusterfs/data/data_subset_selection/subset_data/subset_with_train/KNNSubmod_mnist_cont/KNNSubmod_75.subset",
    #"/glusterfs/data/data_subset_selection/subset_data/subset_with_train/KNNSubmod_mnist_cont/KNNSubmod_50.subset"
    #"/glusterfs/data/data_subset_selection/subset_data/subset_with_train/NBSubmod_mnist/NBSubmod_10.subset",
    #"/glusterfs/data/data_subset_selection/subset_data/subset_with_train/NBSubmod_mnist/NBSubmod_15.subset",
    #"/glusterfs/data/data_subset_selection/subset_data/subset_with_train/NBSubmod_mnist/NBSubmod_20.subset",
    "/glusterfs/data/data_subset_selection/subset_data/subset_with_train/NBSubmod_mnist/NBSubmod_30.subset"
    #"/glusterfs/data/data_subset_selection/subset_data/subset_with_train/NBSubmod_mnist/NBSubmod_50.subset",
    #"/glusterfs/data/data_subset_selection/subset_data/subset_with_train/NBSubmod_mnist/NBSubmod_75.subset"
    #"/glusterfs/data/data_subset_selection/subset_data/subset_with_train/KNNSubmod_cifar100_cont/KNNSubmod_10.subset"
    #"/glusterfs/data/data_subset_selection/subset_data/subset_with_train/KNNSubmod_cifar10_cont/KNNSubmod_10.subset"
    #"/glusterfs/data/data_subset_selection/subset_data/subset_with_train/KNNSubmod_cifar100_cont/KNNSubmod_25.subset"
    #"/glusterfs/data/data_subset_selection/subset_data/subset_with_train/KNNSubmod_cifar10_cont/KNNSubmod_25.subset"
    #"/glusterfs/data/data_subset_selection/subset_data/subset_with_train/KNNSubmod_cifar10_cont/KNNSubmod_75.subset"
]
budgets = [0.3]
run_path = "/home/axd170033/datk-codebase/datk/data-selection/cnnlog/hyperparameter_selection/mnist/"
func = "nbsubmod_normal_exp"
#run_path = "/home/axd170033/datk-codebase/datk/data-selection/cnnlog/hyperparameter_selection/cifar10/"
arguments = ["python", "neural_nets_run.py", "-f", ""]
for j in range(len(datasets)):
    for k in range(len(budgets)):
        for i in range(len(lr)):
            folder = func+"_"+"subset_"+str(int(100*budgets[k]))
            path = run_path+folder
            os.makedirs(path, exist_ok=True)
            input_dict["learning_rate"] = lr[i]
            input_dict["momentum_rate"] = 0.9
            if len(indices_file) > k:
                input_dict["indices_file"] = indices_file[k] 
            input_dict["logfile_path"] = path + "/learning_rate_"+str(int(lr[i]*100))+"_momentum_"+str(int(input_dict["momentum_rate"]*10))+"_resnet18.log"
            input_dict["checkpoint_save_name"] = datasets[j]+"_"+folder+"_"+"learning_rate_"+str(int(lr[i]*100))+"_momentum_"+str(int(input_dict["momentum_rate"]*10))+"_resnet18" 
            input_dict["pytorch_dataset"] = datasets[j]
            input_dict["num_classes"] = num_classes[j]
            input_dict["full_dataset"] = 0
            input_dict["random_subset"] = 0
            input_dict["use_subset"] = 0
            input_dict["current_budget"] = budgets[k] 
            input_dict["is_augmented_data"] = 0
            input_dict["resume"] = 0
            input_dict["num_epoch"] = 200
            input_file = path+"/learning_rate_"+str(int(lr[i]*100))+"_momentum_"+str(int(input_dict["momentum_rate"]*10))+".input"
            cnn_input = open(input_file, "w")
            dict_json = json.dumps(input_dict, indent=4)
            cnn_input.write(dict_json)
            cnn_input.close()
            arguments[3] = input_file
            subprocess.call(arguments)
