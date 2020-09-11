
import datasets_config as dsetconfig
import os
from data_utils import libsvm_to_kNNstandard


## DEPRECATED FUNCTION
def old_write_data_knnsb(datadir, dset_name):
    if dset_name == "dna":
        trn_file = os.path.join(datadir, 'dna.scale.trn')
        val_file = os.path.join(datadir, 'dna.scale.val')
        tst_file = os.path.join(datadir, 'dna.scale.tst')

        data_dims = dsetconfig.dna_config['dims']
        num_cls = dsetconfig.dna_config['num_classes']
        
        libsvm_to_kNNstandard(datadir, trn_file, data_dims)
        libsvm_to_kNNstandard(datadir, val_file, data_dims)
        libsvm_to_kNNstandard(datadir, tst_file, data_dims)

    elif dset_name == "pendigits":
        trn_file = os.path.join(datadir, 'pendigits.trn_full')
        tst_file = os.path.join(datadir, 'pendigits.tst')
        
        data_dims = dsetconfig.pendigits_config['dims']
        num_cls = dsetconfig.pendigits_config['num_classes']

        libsvm_to_kNNstandard(datadir, trn_file, data_dims, split_val=True)        
        libsvm_to_kNNstandard(datadir, tst_file, data_dims)

    elif dset_name == "satimage":
        trn_file = os.path.join(datadir, 'satimage.scale.trn')
        val_file = os.path.join(datadir, 'satimage.scale.val')
        tst_file = os.path.join(datadir, 'satimage.scale.tst')

        data_dims = dsetconfig.satimage_config['dims']
        num_cls = dsetconfig.satimage_config['num_classes']

        libsvm_to_kNNstandard(datadir, trn_file, data_dims)
        libsvm_to_kNNstandard(datadir, val_file, data_dims)
        libsvm_to_kNNstandard(datadir, tst_file, data_dims)

    elif dset_name == "svmguide1":
        trn_file = os.path.join(datadir, 'svmguide1.trn_full')
        tst_file = os.path.join(datadir, 'svmguide1.tst')

        data_dims = dsetconfig.svmguide1_config['dims']
        num_cls = dsetconfig.svmguide1_config['num_classes']

        libsvm_to_kNNstandard(datadir, trn_file, data_dims, split_val=True)        
        libsvm_to_kNNstandard(datadir, tst_file, data_dims)

    elif dset_name == "usps":
        trn_file = os.path.join(datadir, 'usps.trn_full')
        tst_file = os.path.join(datadir, 'usps.tst')

        data_dims = dsetconfig.usps_config['dims']
        num_cls = dsetconfig.usps_config['num_classes']

        libsvm_to_kNNstandard(datadir, trn_file, data_dims, split_val=True)        
        libsvm_to_kNNstandard(datadir, tst_file, data_dims)

    elif dset_name == "ijcnn1":
        trn_file = os.path.join(datadir, 'ijcnn1.trn')
        val_file = os.path.join(datadir, 'ijcnn1.val')
        tst_file = os.path.join(datadir, 'ijcnn1.tst')

        data_dims = dsetconfig.ijcnn1_config['dims']
        num_cls = dsetconfig.ijcnn1_config['num_classes']

        libsvm_to_kNNstandard(datadir, trn_file, data_dims)
        libsvm_to_kNNstandard(datadir, val_file, data_dims)
        libsvm_to_kNNstandard(datadir, tst_file, data_dims)

    elif dset_name == "sklearn-digits":
        data, target = datasets.load_digits(return_X_y=True)
        # Test data is 10%
        x_trn, x_tst, y_trn, y_tst = train_test_split(data, target, test_size=0.1, random_state=42)
        num_cls = 10
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_tst = sc.transform(x_tst)
        fullset = (x_trn, y_trn)
        testset = (x_tst, y_tst)
        return fullset, testset, num_cls
    elif dset_name == "bc":
        data, target = datasets.load_breast_cancer(return_X_y=True)
        x_trn, x_tst, y_trn, y_tst = train_test_split(data, target, test_size=0.1, random_state=42)
        num_cls = 2
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_tst = sc.transform(x_tst)
        fullset = (x_trn, y_trn)
        testset = (x_tst, y_tst)
        return fullset, testset, num_cls






