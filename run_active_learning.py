import subprocess

datadir = sys.argv[1]
data_name = sys.argv[2]

fracs = [0.1, 0.2, 0.3, 0.4, 0.5]
num_epochs = 200
no_of_rounds = [10]  #

feature = ['dss','classimb']

for dset in datasets:
    for sel in no_of_rounds:
        for f in fracs:
            for feat in feature:
                for isOneStepWarm in warm_method:
                    args = ['python3']
                    args.append('active_learning.py')
                    args.append(datadir + dset)
                    args.append(dset)
                    args.append(str(f))
                    args.append(str(num_epochs))
                    args.append(str(sel))
                    args.append(str(feat))
                    print(args)
                    subprocess.run(args)




