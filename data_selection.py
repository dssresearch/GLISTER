'''
    Author : Ayush Dobhal
    Date created : 04/12/2020
    Description : This file contains code to run data selection experiments by running the selection
    schemes. Specify the inputs in the *.input file.
'''
import argparse
import json
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filepath", required=True, type=str, help="json input file")
    arguments = parser.parse_args()
    input_file = arguments.filepath
    with open(input_file) as f:
      input_dict = json.load(f)
   
    dataset_trn = input_dict["dataset_train"]
    dataset_val = input_dict["dataset_validation"]
    dataset_tst = input_dict["dataset_test"]
    executables = input_dict["executables"]
    budgets = input_dict["budgets"]
    dataset_type = input_dict["dataset_type"]
    dataset_names = input_dict["dataset_names"]
    run_path = input_dict["run_path"]
    executable_name = ["NBSubmod"]
    
    # sample arugments list. Modifying it in the for loop based on the dataset and other args.
    arguments = ["/home/axd170033/datk-codebase/datk/build/knn_submod", "/glusterfs/data/dataset/discrete_datasets/car/trf_car.data.trn", "/glusterfs/data/dataset/discrete_datasets/car/trf_car.data.val", " ", "0.25", run_path, "0"]
    
    print("Running all experiments")
    
    for k in range(len(dataset_trn)):
        for i in range(len(executables)):
            subset_model_accuracy = []
            random_model_accuracy = []
            if "NBSubmod" == executable_name[i] and dataset_type[k] == "continuous":
                continue
            #log_file = open(executable_name[i]+"_"+dataset_names[k]+".log", "a")
            output_path = run_path + executable_name[i]+"_"+dataset_names[k] + "/"
            subprocess.call(["mkdir", output_path])
            for j in range(len(budgets)):
                #log_file = open(executable_name[i]+"_"+dataset_names[k]+".log", "a")
                arguments[0] = executables[i]
                arguments[1] = dataset_trn[k]
                arguments[2] = dataset_val[k]
                arguments[4] = str(budgets[j])
                arguments[5] = output_path + executable_name[i]+"_"+str((int)(budgets[j]*100))+".subset"
                if dataset_type[k] == "continuous":
                    arguments[6] = "1"
    
                print("Obtaining the subset")
                subprocess.run(arguments)

if __name__ == "__main__":
    main()





""" Storing the sample data_selection.input below as a comment:


{
    "dataset_train" : [
      "/glusterfs/data/dataset/discrete_datasets/car/trf_car.data.trn",
      "/glusterfs/data/dataset/image_binned_datasets/cifar10/cifar10_bin.trn",
      "/glusterfs/data/dataset/image_binned_datasets/cifar100/cifar100_bin.trn",
      "/glusterfs/data/dataset/image_binned_datasets/cifar10_continuous/cifar10.trn",
      "/glusterfs/data/dataset/image_binned_datasets/cifar100_continuous/cifar100.trn"
    ],  
    "dataset_validation" : [
      "/glusterfs/data/dataset/discrete_datasets/car/trf_car.data.trn",
      "/glusterfs/data/dataset/image_binned_datasets/cifar10/cifar10_bin.trn",
      "/glusterfs/data/dataset/image_binned_datasets/cifar100/cifar100_bin.trn",
      "/glusterfs/data/dataset/image_binned_datasets/cifar10_continuous/cifar10.trn",
      "/glusterfs/data/dataset/image_binned_datasets/cifar100_continuous/cifar100.trn"

    ], 
    "dataset_test" : [" "
    ],
    "budgets" : [0.2, 0.4, 0.6],
    "executables" : [
        "/home/axd170033/datk-codebase/datk/build/KNNSubmod", 
        "/home/axd170033/datk-codebase/datk/build/NBSubmod"
    ],
    "dataset_type" : [
        "discrete",
        "discrete",
        "discrete",
        "continuous",
        "continuous"
    ],
    "dataset_names" : [
        "cars",
        "cifar10_bin",
        "cifar100_bin",
        "cifar10_cont",
        "cifar100_cont"
    ],
    "run_path" : "/home/axd170033/datk-codebase/datk/data-selection/run_data_full/"
}

"""
