from sklearn import datasets
import numpy as np
import apricot
import time


def find_indices(data, data_sub):
    indices = []
    for ele in data_sub:
        x = np.where((data == ele).all(axis=1))[0]
        indices.append(x[0])
    return indices


num_samples = 100000
num_centers = 2
num_features = 5

X, y= datasets.make_blobs(n_samples=num_samples, centers=num_centers, n_features=num_features,center_box=(-8.75, 8.75))
#center_box=(-8.75, 8.75))

start_time = time.time()
fl = apricot.functions.facilityLocation.FacilityLocationSelection(random_state=0, n_samples=1000)
X_sub = fl.fit_transform(X)
end_time = time.time()
facloc_indices = find_indices(X, X_sub)

print("Time taken for running facility location is " + str(end_time-start_time))