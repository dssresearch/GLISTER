import subprocess

datadir = './data/'
#datasets = ['sklearn-digits', 'dna','satimage','svmguide1']
# datasets = ['dna']
datasets = ['covertype']
# datasets = ['ijcnn1']
fracs = [0.1, 0.2, 0.3, 0.4, 0.5]
num_epochs = 200
select_every = [10]  # ,35,50]
warm_method = [0]  # 0 = online, 1 = onestep warmstart
num_runs = 10
#feature = ['dss']  # ,'noise','classimb']
# feature = ['noise']
feature = ['classimb']
for dset in datasets:
    for sel in select_every:
        for f in fracs:
            for feat in feature:
                for isOneStepWarm in warm_method:
                    args = ['python3']
                    args.append('active_learning.py')
                    if dset in ['mnist', "fashion-mnist"]:
                        args.append(datadir + dset.upper())
                    else:
                        args.append(datadir + dset)
                    args.append(dset)
                    args.append(str(f))
                    args.append(str(num_epochs))
                    args.append(str(sel))
                    args.append(str(feat))
                    args.append(str(isOneStepWarm))
                    args.append(str(num_runs))
                    print(args)
                    subprocess.run(args)




