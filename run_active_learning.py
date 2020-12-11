import subprocess

#datadir = sys.argv[1]
#data_name = sys.argv[2]
<<<<<<< HEAD
#datadir = "C:\\Users\\krish\\OneDrive - The University of Texas at Dallas\\Documents\\GitHub\\GLISTER\\data\\"
datadir = "../Datasets/data/"
fracs = [0.1, 0.2, 0.3, 0.4, 0.5]
num_epochs = 200
no_of_rounds = [10]  #
datasets = ['sklearn-digits', 'dna','satimage','svmguide1']
feature = ['dss', 'classimb']
=======
datadir = "../data/"
fracs = [0.1, 0.2, 0.3]# 0.4, 0.5]
num_epochs = 200
no_of_rounds = [10]  #
datasets = ['sklearn-digits', 'dna', 'svmguide1', 'satimage']
feature = ['dss']
>>>>>>> c59fb2f7533c4623dc946a71c883ceb36e371ae0

for dset in datasets:
    for sel in no_of_rounds:
        for f in fracs:
            for feat in feature:
                    args = ['python']
                    args.append('active_learning.py')
                    args.append(datadir + dset)
                    args.append(dset)
                    args.append(str(f))
                    args.append(str(num_epochs))
                    args.append(str(sel))
                    args.append(str(feat))
                    print(args)
                    subprocess.run(args)




