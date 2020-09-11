import subprocess

datadir = './data/'
datasets = ['mnist']#['large_linsep_4', 'clf_2', 'linsep', 'linsep_4', 'gauss_2']#['sklearn-digits', 'dna', 'satimage', 'svmguide1', 'pendigits', 'usps']
fracs = [0.1, 0.2, 0.3, 0.4, 0.5]
num_epochs = 100
feature = ['classimb']
select_every = [20]#, 35, 50]
# select_every = [20, 60, 100]
warm_method = [0]   # 0 = online, 1 = onestep warmstart
num_runs = 10
for dset in datasets:
    for sel in select_every:
        for f in fracs:
            for isOneStepWarm in warm_method:
                for feat in feature:
                    args = ['python']
                    args.append('mnist_classimb.py')
                    #args.append('new_run_onestep_selection_minibatch.py') # selection every few!
                    #args.append('run_knnsubmod_selection_fullbatch.py') # selection using KNNsubmod indices
                    args.append(datadir + dset)
                    args.append(dset)
                    args.append(str(f))
                    args.append(str(num_epochs))
                    args.append(str(sel))
                    args.append(feat)
                    args.append(str(isOneStepWarm))
                    args.append(str(num_runs))
                    print(args)
                    subprocess.run(args)




