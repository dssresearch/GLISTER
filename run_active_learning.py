import subprocess

datadir = './data/'
#datadir = '../../data/'
#datasets = ['large_linsep_4', 'clf_2', 'linsep', 'linsep_4', 'gauss_2']#['sklearn-digits', 'dna', 'satimage', 'svmguide1', 'pendigits', 'usps']
#datasets = ['prior_shift_large_linsep_4', 'prior_shift_clf_2', 'prior_shift_gauss_2','conv_shift_large_linsep_4', 'conv_shift_clf_2', 'conv_shift_gauss_2']
#datasets = ['shrink_large_linsep_4','expand_large_linsep_4']
#datasets = ['red_conv_shift_large_linsep_4','red_large_linsep_4']
#datasets = ['sklearn-digits', 'dna'] 
datasets = ['satimage']
#datasets = [ 'svmguide1']
#datasets =['letter','shuttle','ijcnn1','sensorless','connect_4','sensit_seismic']

fracs = [ 0.1, 0.2,0.4]
num_epochs = 200
num_runs = 10
imbalance = True

for dset in datasets:
    for f in fracs:
        args = ['python3']
        #args.append('run_onestep_selection_fullbatch.py') # selection every few!
        args.append('active_learning.py')
        ''' args = ['python']
        args.append('new_run_onestep_selection_minibatch.py') # selection every few!
        #args.append('run_knnsubmod_selection_fullbatch.py') # selection using KNNsubmod indices'''
        args.append(datadir + dset)
        args.append(dset)
        args.append(str(f))
        args.append(str(num_epochs))
        args.append(str(num_runs))
        args.append(str(imbalance))
        print(args)
        subprocess.run(args)
