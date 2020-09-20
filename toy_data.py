import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
import random
import subprocess

def diff_class_prior(X,y,num_centers):

    train_indices = []
    val_indices = []
    tst_indices = []

    choice = [x / 100.0 for x in range(2,10)]

    frac = list(np.random.choice(choice, size=int(num_centers/2), replace=True))

    for i in range(int(num_centers/2)):
        frac.append(0.2-frac[i])

    if num_centers %2 ==1:
        frac.append(0.1)

    random.shuffle(frac)

    for i in range(num_centers):
        temp_indices = (set(np.where(y == i)[0]))
        val_indices.extend(np.random.choice(list(temp_indices), size=int((frac[i]/num_centers) * len(y)), replace=False))
        temp_indices = temp_indices.difference(val_indices)
        tst_indices.extend(np.random.choice(list(temp_indices), size=int((frac[i]*2/num_centers) * len(y)), replace=False))
        temp_indices = temp_indices.difference(tst_indices)
        train_indices.extend(list(temp_indices))

    return train_indices, val_indices, tst_indices


def generate_linear_seperable_data(num_samples, num_centers, num_features, file_name,same=None, noise_ratio = 0, probability = 1):
    X, y, centers= datasets.make_blobs(n_samples=num_samples, centers=num_centers,
                                       n_features=num_features, return_centers=True,center_box=(-5.75, 5.75))
    #center_box=(-8.75, 8.75))
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=y,
                s=25, edgecolor='k')
    plt.show()
    train_indices = []
    val_indices = []
    tst_indices = []

    shift = np.array([0,0])
    min_cnt = np.min(num_samples)
    if same == "class_imb":
        for i in range(len(num_samples)):
            temp_indices = (set(np.where(y == i)[0]))
            val_indices.extend(
                np.random.choice(list(temp_indices), size=int(0.4 * min_cnt), replace=False))
            temp_indices = temp_indices.difference(val_indices)
            tst_indices.extend(
                np.random.choice(list(temp_indices), size=int(0.4 * min_cnt), replace=False))
            temp_indices = temp_indices.difference(tst_indices)
            train_indices.extend(list(temp_indices))
    else:
        for i in range(num_centers):
            temp_indices = (set(np.where(y == i)[0]))
            val_indices.extend(np.random.choice(list(temp_indices), size=int((0.1/num_centers) * len(y)), replace=False))
            temp_indices = temp_indices.difference(val_indices)
            tst_indices.extend(np.random.choice(list(temp_indices), size=int((0.2/num_centers) * len(y)), replace=False))
            temp_indices = temp_indices.difference(tst_indices)
            train_indices.extend(list(temp_indices))

        if same == "covariate":
            choice = [x / 10.0 for x in range(-30,30)]
            remove = [x / 10.0 for x in range(-14,15)]
            choice = list(set(choice).difference(remove))

            shift = np.random.choice(choice, size=2, replace=True)

        elif same == "expand":
            factor = np.random.choice([i for i in range(2,5)], size=1, replace=True)
            #factor = 2
            print(factor)
            for i in val_indices+tst_indices:
                curr_center = centers[y[i]]
                X[i] = factor*X[i] - (factor-1)*curr_center         

        elif same == "shrink":
            factor = np.random.choice([i for i in range(2,5)], size=1, replace=True)
            print(factor)
            for i in val_indices+tst_indices:
                curr_center = centers[y[i]]
                X[i] = ((factor-1)*curr_center + X[i])/factor


    data_dir = "./data/"+file_name
    subprocess.run(["mkdir", data_dir])


    if same == "noise":
        noise_size = int(len(train_indices) * noise_ratio)
        noise_indices = np.random.choice(list(train_indices), size=noise_size, replace=False)
        y[noise_indices] = np.random.choice(np.arange(num_centers), size=noise_size, replace=True)



    X_train = X[train_indices]
    y_train = y[train_indices]

    X_tot_train = np.concatenate((X_train, y_train.reshape((-1, 1))), axis=1)
    plt.scatter(X_train[:, 0], X_train[:, 1], marker='o', c=y_train,
                s=25, edgecolor='k')
    plt.savefig(data_dir + "/training_data.png")
    plt.show()

    X_val = X[val_indices] + shift
    y_val = y[val_indices]
    print(shift)
    X_tot_val = np.concatenate((X_val, y_val.reshape((-1, 1))), axis=1)
    plt.figure()
    plt.scatter(X_val[:, 0], X_val[:, 1], marker='o', c=y_val,
                s=25, edgecolor='k')
    plt.savefig(data_dir + "/validation_data.png")
    plt.show()

    X_tst = X[tst_indices] + shift
    y_tst = y[tst_indices]
    X_tot_tst = np.concatenate((X_tst, y_tst.reshape((-1, 1))), axis=1)
    plt.figure()
    plt.scatter(X_tst[:, 0], X_tst[:, 1], marker='o', c=y_tst,
                s=25, edgecolor='k')
    plt.savefig(data_dir + "/test_data.png")
    plt.show()

    with open(data_dir+"/"+file_name + ".trn", 'w') as f:
        np.savetxt(f, X_tot_train, delimiter=",")
    with open(data_dir+"/"+file_name + ".val", 'w') as f:
        np.savetxt(f, X_tot_val, delimiter=",")
    with open(data_dir+"/"+file_name + ".tst", 'w') as f:
        np.savetxt(f, X_tot_tst, delimiter=",")

#generate_linear_seperable_data(5000, 4, 2, 'red_large_linsep_4')
#generate_linear_seperable_data(10000, 4, 2, 'prior_shift_large_linsep_4',"prior")
#generate_linear_seperable_data(10000, 4, 2, 'conv_shift_large_linsep_4',"covariate")
#generate_linear_seperable_data(5000, 4, 2, 'expand_large_linsep_4',"expand")
#generate_linear_seperable_data(num_samples = np.array([50, 500, 50, 500]), num_centers=None, num_features=2, file_name='class_imb_linsep_4', same='class_imb')

def generate_gaussian_quantiles_data(num_samples, num_centers, num_features, file_name,same=None):
    #num_labels = int(num_samples)/num_features
    X, y, centers = datasets.make_gaussian_quantiles(mean=None, cov=1.0, n_samples=num_samples, n_features=2, n_classes=num_centers, shuffle=True, random_state=None)
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=y,
                s=25, edgecolor='k')
    plt.show()
    train_indices = []
    val_indices = []
    tst_indices = []

    shift = np.array([0,0])

    if same == "prior":
        train_indices, val_indices, tst_indices = diff_class_prior(X,y,num_centers)
    else:
        for i in range(num_centers):
            temp_indices = (set(np.where(y == i)[0]))
            val_indices.extend(np.random.choice(list(temp_indices), size=int((0.1/num_centers) * len(y)), replace=False))
            temp_indices = temp_indices.difference(val_indices)
            tst_indices.extend(np.random.choice(list(temp_indices), size=int((0.2/num_centers) * len(y)), replace=False))
            temp_indices = temp_indices.difference(tst_indices)
            train_indices.extend(list(temp_indices))

        if same == "covariate":
            choice = [x / 10.0 for x in range(-30,30)]
            remove = [x / 10.0 for x in range(-9,10)]
            choice = list(set(choice).difference(remove))

            shift = np.random.choice(choice, size=2, replace=True)

        elif same == "expand":
            factor = np.random.choice([i for i in range(2,5)], size=1, replace=True)
            print(factor)
            for i in val_indices+tst_indices:
                curr_center = centers[y[i]]
                X[i] = factor*X[i] - (factor-1)*curr_center         

        elif same == "shrink":
            factor = np.random.choice([i for i in range(2,5)], size=1, replace=True)
            print(factor)
            for i in val_indices+tst_indices:
                curr_center = centers[y[i]]
                X[i] = ((factor-1)*curr_center + X[i])/factor
    
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_tot_train = np.concatenate((X_train, y_train.reshape((-1, 1))), axis=1)
    plt.scatter(X_train[:, 0], X_train[:, 1], marker='o', c=y_train,
                s=25, edgecolor='k')
    plt.show()
    X_val = X[val_indices]
    y_val = y[val_indices]
    X_tot_val = np.concatenate((X_val, y_val.reshape((-1, 1))), axis=1)
    plt.scatter(X_val[:, 0], X_val[:, 1], marker='o', c=y_val,
                s=25, edgecolor='k')
    plt.show()
    X_tst = X[tst_indices]
    y_tst = y[tst_indices]
    X_tot_tst = np.concatenate((X_tst, y_tst.reshape((-1, 1))), axis=1)
    plt.scatter(X_tst[:, 0], X_tst[:, 1], marker='o', c=y_tst,
                s=25, edgecolor='k')
    plt.show()

    data_dir = "./data/"+file_name
    subprocess.run(["mkdir", data_dir])

    with open(data_dir+"/"+file_name + ".trn", 'w') as f:
        np.savetxt(f, X_tot_train, delimiter=",")
    with open(data_dir+"/"+file_name + ".val", 'w') as f:
        np.savetxt(f, X_tot_val, delimiter=",")
    with open(data_dir+"/"+file_name + ".tst", 'w') as f:
        np.savetxt(f, X_tot_tst, delimiter=",")

#generate_gaussian_quantiles_data(10000, 2, 2, 'gauss_2')
#generate_gaussian_quantiles_data(10000, 2, 2, 'prior_shift_gauss_2',"prior")
#generate_gaussian_quantiles_data(10000, 2, 2, 'conv_shift_gauss_2',"convariate")

def generate_classification_data(num_samples, num_centers, file_name,same=None):
    #num_labels = int(num_samples)/num_features
    X, y= datasets.make_classification(n_samples=num_samples, n_features=2, n_redundant=0, n_informative=2, n_classes=num_centers,
                                        n_clusters_per_class=1) #, centers 
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=y,
                s=25, edgecolor='k')
    plt.show()
    train_indices = []
    val_indices = []
    tst_indices = []

    shift = np.array([0,0])

    if same == "prior":
        train_indices, val_indices, tst_indices = diff_class_prior(X,y,num_centers)
    else:
        for i in range(num_centers):
            temp_indices = (set(np.where(y == i)[0]))
            val_indices.extend(np.random.choice(list(temp_indices), size=int((0.1/num_centers) * len(y)), replace=False))
            temp_indices = temp_indices.difference(val_indices)
            tst_indices.extend(np.random.choice(list(temp_indices), size=int((0.2/num_centers) * len(y)), replace=False))
            temp_indices = temp_indices.difference(tst_indices)
            train_indices.extend(list(temp_indices))

        if same == "covariate":
            choice = [x / 10.0 for x in range(-30,30)]
            remove = [x / 10.0 for x in range(-9,10)]
            choice = list(set(choice).difference(remove))

            shift = np.random.choice(choice, size=2, replace=True)

        elif same == "expand":
            factor = np.random.choice([i for i in range(2,5)], size=1, replace=True)
            print(factor)
            for i in val_indices+tst_indices:
                curr_center = centers[y[i]]
                X[i] = factor*X[i] - (factor-1)*curr_center         

        elif same == "shrink":
            factor = np.random.choice([i for i in range(2,5)], size=1, replace=True)
            print(factor)
            for i in val_indices+tst_indices:
                curr_center = centers[y[i]]
                X[i] = ((factor-1)*curr_center + X[i])/factor
    
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_tot_train = np.concatenate((X_train, y_train.reshape((-1, 1))), axis=1)
    plt.scatter(X_train[:, 0], X_train[:, 1], marker='o', c=y_train,
                s=25, edgecolor='k')
    plt.show()
    X_val = X[val_indices] +shift
    y_val = y[val_indices]
    X_tot_val = np.concatenate((X_val, y_val.reshape((-1, 1))), axis=1)
    plt.scatter(X_val[:, 0], X_val[:, 1], marker='o', c=y_val,
                s=25, edgecolor='k')
    plt.show()
    X_tst = X[tst_indices] +shift
    y_tst = y[tst_indices] 
    X_tot_tst = np.concatenate((X_tst, y_tst.reshape((-1, 1))), axis=1)
    plt.scatter(X_tst[:, 0], X_tst[:, 1], marker='o', c=y_tst,
                s=25, edgecolor='k')
    plt.show()

    data_dir = "./data/"+file_name
    subprocess.run(["mkdir", data_dir])

    with open(data_dir+"/"+file_name + ".trn", 'w') as f:
        np.savetxt(f, X_tot_train, delimiter=",")
    with open(data_dir+"/"+file_name + ".val", 'w') as f:
        np.savetxt(f, X_tot_val, delimiter=",")
    with open(data_dir+"/"+file_name + ".tst", 'w') as f:
        np.savetxt(f, X_tot_tst, delimiter=",")


#generate_classification_data(10000, 2, 'clf_2')
#generate_classification_data(10000, 2, 'prior_shift_clf_2','prior')
generate_classification_data(10000, 2, 'conv_shift_clf_2','covariate')