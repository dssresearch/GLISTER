import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from utils import custom_dataset

datadir = './data/'
resultsdir = './results/LR/'
#datasets = ['large_linsep_4', 'clf_2', 'linsep', 'linsep_4', 'gauss_2']#['sklearn-digits', 'dna', 'satimage', 'svmguide1', 'pendigits', 'usps']
#datasets = ['prior_shift_large_linsep_4', 'prior_shift_clf_2', 'prior_shift_gauss_2','conv_shift_large_linsep_4', 'conv_shift_clf_2', 'conv_shift_gauss_2']
datasets = ['class_imb_linsep_4','class_imb_linsep_2']
fracs = [0.05, 0.1, 0.2]# 0.2]
select_every = [20]#[20, 35, 50, 65]

for f in fracs:
    for sel in select_every:
        for dset in datasets:
            data_dir = resultsdir + dset + "/" + str(f) + "/" + str(sel) + "/"
            file_name = dset + '.trn'
            data_file = datadir + dset + "/" + file_name
            val_file = datadir + dset + "/" + dset + '.val'
            one_step_subset_file = data_dir + "one_step_subset_selected.txt"
            without_taylor_subset_file = data_dir + "without_taylor_subset_selected.txt"
            taylor_logit_subset_file = data_dir + "taylor_logit_subset_selected.txt"
            step_one_step_subset_file = data_dir + "stepwise_one_step_subset_selected.txt"
            mod_one_step_subset_file = data_dir + "mod_one_step_subset_selected.txt"
            facloc_subset_file = data_dir + "facloc_subset_selected.txt"
            random_subset_file = data_dir + "random_subset_selected.txt"
            rand_one_step_subset_file = data_dir + "rand_reg_one_step_subset_selected.txt"
            facloc_reg_one_step_subset_file = data_dir + "facloc_reg_one_step_subset_selected.txt"

            X, y = custom_dataset.libsvm_file_load(data_file, dim=2)

            X_val, y_val = custom_dataset.libsvm_file_load(val_file, dim=2)

            with open(one_step_subset_file, 'r') as read_file:
                txt = read_file.readlines()
                txt = txt[0].split("[")[1]
                txt = txt.split("]")[0]
                one_step_subset_idx = [int(x) for x in txt.split(",")]


            #with open(without_taylor_subset_file, 'r') as read_file:
            #    txt = read_file.readlines()
            #    txt = txt[0].split("[")[1]
            #    txt = txt.split("]")[0]
            #    without_taylor_subset_idx = [int(x) for x in txt.split(",")]

            #with open(taylor_logit_subset_file, 'r') as read_file:
            #    txt = read_file.readlines()
            #    txt = txt[0].split("[")[1]
            #    txt = txt.split("]")[0]
            #    taylor_logit_subset_idx = [int(x) for x in txt.split(",")]

            with open(step_one_step_subset_file, 'r') as read_file:
                txt = read_file.readlines()
                txt = txt[0].split("[")[1]
                txt = txt.split("]")[0]
                step_one_step_subset_idx = [int(x) for x in txt.split(",")]

            with open(rand_one_step_subset_file, 'r') as read_file:
                txt = read_file.readlines()
                txt = txt[0].split("[")[1]
                txt = txt.split("]")[0]
                rand_one_step_subset_idx = [int(x) for x in txt.split(",")]

            with open(facloc_reg_one_step_subset_file, 'r') as read_file:
                txt = read_file.readlines()
                txt = txt[0].split("[")[1]
                txt = txt.split("]")[0]
                facloc_reg_one_step_subset_idx = [int(x) for x in txt.split(",")]

            #with open(mod_one_step_subset_file, 'r') as read_file:
            #    txt = read_file.readlines()
            #    txt = txt[0].split("[")[1]
            #    txt = txt.split("]")[0]
            #    mod_one_step_subset_idx = [int(x) for x in txt.split(",")]

            with open(facloc_subset_file, 'r') as read_file:
                txt = read_file.readlines()
                txt = txt[0].split("[")[1]
                txt = txt.split("]")[0]
                facloc_subset_idx = [int(x) for x in txt.split(",")]

            with open(random_subset_file, 'r') as read_file:
                txt = read_file.readlines()
                txt = txt[0].split("[")[1]
                txt = txt.split("]")[0]
                random_subset_idx = [int(x) for x in txt.split(",")]


            plt.figure()
            plt.scatter(X[:, 0], X[:, 1], marker='o', c=y,
                            s=25, edgecolor='k')
            plt.title("Training Data")
            plt.savefig(data_dir + "training_data.png")
            #plt.show()

            '''plt.figure()
            plt.scatter(X_val[:, 0], X_val[:, 1], marker='o', c=y_val,
                            s=25, edgecolor='k')
            plt.title("Validation Data")
            plt.savefig(data_dir + "val_data.png")'''
            #plt.show()

            plt.figure()
            plt.scatter(X[:, 0], X[:, 1], marker='o', c=y,
                            s=25, edgecolor='k')
            plt.scatter(X_val[:, 0], X_val[:, 1], marker='o', c='r',
                            s=25, edgecolor='k')
            plt.title("Validation Hightlighted with Training Data")
            plt.savefig(data_dir + "val_data.png")

            plt.figure()
            plt.scatter(X[:, 0], X[:, 1], marker='o', c=y,
                            s=25, edgecolor='k')
            plt.scatter(X[one_step_subset_idx][:, 0], X[one_step_subset_idx][:, 1], marker='o', c='r',
                           s=25, edgecolor='k')
            plt.title("One Step Hightlight Training Data")
            plt.savefig(data_dir + "one_step_highlight_training_data.png")
            #plt.show()

            #plt.figure()
            #plt.scatter(X[:, 0], X[:, 1], marker='o', c=y,
            #                s=25, edgecolor='k')
            #plt.scatter(X[without_taylor_subset_idx][:, 0], X[without_taylor_subset_idx][:, 1], marker='o', c='r',
            #               s=25, edgecolor='k')
            #plt.title("Full one step without taylor Hightlight Training Data")
            #plt.savefig(data_dir + "without_taylor_highlight_training_data.png")

           # plt.figure()
           # plt.scatter(X[:, 0], X[:, 1], marker='o', c=y,
            #                s=25, edgecolor='k')
            #plt.scatter(X[taylor_logit_subset_idx][:, 0], X[taylor_logit_subset_idx][:, 1], marker='o', c='r',
            #               s=25, edgecolor='k')
            #plt.title("Taylor on logit Hightlight Training Data")
            #plt.savefig(data_dir + "taylor_logit_highlight_training_data.png")

            plt.figure()
            plt.scatter(X[:, 0], X[:, 1], marker='o', c=y,
                            s=25, edgecolor='k')
            plt.scatter(X[step_one_step_subset_idx][:, 0], X[step_one_step_subset_idx][:, 1], marker='o', c='r',
                           s=25, edgecolor='k')
            plt.title("One Step stepwise Hightlight Training Data")
            plt.savefig(data_dir + "stepwise_one_step_highlight_training_data.png")
            #plt.show()

            #plt.figure()
            #plt.scatter(X[:, 0], X[:, 1], marker='o', c=y,
            #                s=25, edgecolor='k')
            #plt.scatter(X[mod_one_step_subset_idx][:, 0], X[mod_one_step_subset_idx][:, 1], marker='o', c='r',
            #               s=25, edgecolor='k')
            #plt.title("Mod One Step Hightlight Training Data")
            #plt.savefig(data_dir + "mod_one_step_highlight_training_data.png")
            #plt.show()

            plt.figure()
            plt.scatter(X[:, 0], X[:, 1], marker='o', c=y,
                            s=25, edgecolor='k')
            plt.scatter(X[rand_one_step_subset_idx][:, 0], X[rand_one_step_subset_idx][:, 1], marker='o', c='r',
                           s=25, edgecolor='k')
            plt.title("Rand Reg One Step Hightlight Training Data")
            plt.savefig(data_dir + "rand_reg_one_step_highlight_training_data.png")
            #plt.show()

            plt.figure()
            plt.scatter(X[:, 0], X[:, 1], marker='o', c=y,
                            s=25, edgecolor='k')
            plt.scatter(X[facloc_reg_one_step_subset_idx][:, 0], X[facloc_reg_one_step_subset_idx][:, 1], marker='o', c='r',
                           s=25, edgecolor='k')
            plt.title("Facloc Reg One Step Hightlight Training Data")
            plt.savefig(data_dir + "facloc_reg_one_step_highlight_training_data.png")
            #plt.show()

            plt.figure()
            plt.scatter(X[:, 0], X[:, 1], marker='o', c=y,
                            s=25, edgecolor='k')
            plt.scatter(X[facloc_subset_idx][:, 0], X[facloc_subset_idx][:, 1], marker='o', c='r',
                           s=25, edgecolor='k')
            plt.title("FacLoc Hightlight Training Data")
            plt.savefig(data_dir + "facloc_highlight_training_data.png")
            #plt.show()

            plt.figure()
            plt.scatter(X[:, 0], X[:, 1], marker='o', c=y,
                            s=25, edgecolor='k')
            plt.scatter(X[random_subset_idx][:, 0], X[random_subset_idx][:, 1], marker='o', c='r',
                           s=25, edgecolor='k')
            plt.title("Random Hightlight Training Data")
            plt.savefig(data_dir + "random_highlight_training_data.png")
            #plt.show()

            plt.figure()
            plt.scatter(X[one_step_subset_idx][:, 0], X[one_step_subset_idx][:, 1], marker='o', c=y[one_step_subset_idx],
                           s=25, edgecolor='k')
            plt.title("One Step Subset Selected")
            plt.savefig(data_dir + "one_step_subset_selected.png")
            #plt.show()

            plt.figure()
            plt.scatter(X[step_one_step_subset_idx][:, 0], X[step_one_step_subset_idx][:, 1], marker='o', c=y[step_one_step_subset_idx],
                           s=25, edgecolor='k')
            plt.title("One Step stepwise Hightlight Training Data")
            plt.savefig(data_dir + "stepwise_one_step_selected.png")
            #plt.show()

            #plt.figure()
            #plt.scatter(X[mod_one_step_subset_idx][:, 0], X[mod_one_step_subset_idx][:, 1], marker='o', c=y[mod_one_step_subset_idx],
            #               s=25, edgecolor='k')
            #plt.title("Mod One Step Subset Selected")
            #plt.savefig(data_dir + "mod_one_step_subset_selected.png")
            #plt.show()

            plt.figure()
            plt.scatter(X[rand_one_step_subset_idx][:, 0], X[rand_one_step_subset_idx][:, 1], marker='o', c=y[rand_one_step_subset_idx],
                           s=25, edgecolor='k')
            plt.title("Rand Reg One Step Subset Selected")
            plt.savefig(data_dir + "rand_one_step_subset_selected.png")
            #plt.show()

            plt.figure()
            plt.scatter(X[facloc_reg_one_step_subset_idx][:, 0], X[facloc_reg_one_step_subset_idx][:, 1], marker='o', c=y[facloc_reg_one_step_subset_idx],
                           s=25, edgecolor='k')
            plt.title("Facloc reg One Step Subset Selected")
            plt.savefig(data_dir + "facloc_reg_one_step_subset_selected.png")
            #plt.show()

            plt.figure()
            plt.scatter(X[facloc_subset_idx][:, 0], X[facloc_subset_idx][:, 1], marker='o', c=y[facloc_subset_idx],
                           s=25, edgecolor='k')
            plt.title("Facility Location Subset Selected")
            plt.savefig(data_dir + "facloc_subset_selected.png")
            #plt.show()

            plt.figure()
            plt.scatter(X[random_subset_idx][:, 0], X[random_subset_idx][:, 1], marker='o', c=y[random_subset_idx],
                           s=25, edgecolor='k')
            plt.title("Random Subset Selected")
            plt.savefig(data_dir + "random_subset_selected.png")
            #plt.show()