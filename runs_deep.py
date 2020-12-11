import subprocess
import sys

#datadir = sys.argv[1]
#data_name = sys.argv[2]
datadir = '../data/'
fracs =[0.1]
num_epochs = 3
select_every = [2]#,35,50]
warm_method = [0]   # 0 = online, 1 = onestep warmstart
num_runs = 10
feature = ['dss']
datasets = ['mnist']

for dset in datasets:
    for sel in select_every:
        for f in fracs:
            for feat in feature:
                for isOneStepWarm in warm_method:
                    args = ['python']
                    args.append('dss_deep.py')
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