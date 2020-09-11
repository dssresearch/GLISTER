import subprocess

#datadir = 'C:\\Users\\krish\\OneDrive - The University of Texas at Dallas\\Documents\\datk\\data-selection\\data\\'
datadir = './data/'
datasets = ['mnist']
fracs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
num_epochs = 150
select_every = [1]#, 35, 50]
warm_method = [0]   # 0 = online, 1 = onestep warmstart
num_runs = 10
noise_fracs = [0.3, 0.5, 0.7]
for dset in datasets:
    for sel in select_every:
        for f in fracs:
            for noise_frac in noise_fracs:
                for isOneStepWarm in warm_method:
                    args = ['python']
                    args.append('mnist_noise.py')
                    args.append(datadir + dset)
                    args.append(dset)
                    args.append(str(f))
                    args.append(str(num_epochs))
                    args.append(str(sel))
                    args.append(str(noise_frac))
                    args.append(str(isOneStepWarm))
                    args.append(str(num_runs))
                    print(args)
                    subprocess.run(args)




