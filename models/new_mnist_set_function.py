import numpy as np
import time
import torch
import math
import torch.nn.functional as F
from torch.utils.data import SequentialSampler, BatchSampler
from queue import PriorityQueue
from torch import random


class SetFunctionLoader_2(object):
    def __init__(self, trainset, x_val, y_val, model, loss_criterion,
                 loss_nored, eta, device, num_classes, batch_size):
        self.trainset = trainset  # assume its a sequential loader.
        self.x_val = x_val.to(device)
        self.y_val = y_val.to(device)
        self.model = model
        self.loss = loss_criterion  # Make sure it has reduction='none' instead of default
        self.loss_nored = loss_nored
        self.eta = eta  # step size for the one step gradient update
        # self.opt = optimizer
        self.device = device
        self.N_trn = len(trainset)
        self.grads_per_elem = None
        self.theta_init = None
        self.num_classes = num_classes
        self.batch_size = batch_size

    def _compute_per_element_grads(self, theta_init):
        self.model.load_state_dict(theta_init)
        batch_wise_indices = np.array(
            [list(BatchSampler(SequentialSampler(np.arange(self.N_trn)), self.batch_size, drop_last=False))][0])
        cnt = 0
        for batch_idx in batch_wise_indices:
            inputs = torch.cat(
                [self.trainset[x][0].view(-1, 1, self.trainset[x][0].shape[1], self.trainset[x][0].shape[2]) for x in
                 batch_idx], dim=0).type(torch.float)
            targets = torch.tensor([self.trainset[x][1] for x in batch_idx])
            inputs, targets = inputs.to(self.device), targets.to(self.device, non_blocking=True)
            if cnt == 0:
                with torch.no_grad():
                    out, l1 = self.model(inputs)
                    data = F.softmax(out, dim=1)
                tmp_tensor = torch.zeros(len(inputs), self.num_classes).to(self.device)
                tmp_tensor.scatter_(1, targets.view(-1, 1), 1)
                outputs = tmp_tensor
                l0_grads = data - outputs
                l1_grads = torch.zeros(self.batch_size, self.num_classes, l1.shape[1]).to(self.device)
                for i in range(self.batch_size):
                    for j in range(self.num_classes):
                        l1_grads[i, j, :] = l0_grads[i, j] * l1[i]
                cnt = cnt + 1
            else:
                with torch.no_grad():
                    out, l1 = self.model(inputs)
                    data = F.softmax(out, dim=1)
                tmp_tensor = torch.zeros(len(inputs), self.num_classes).to(self.device)
                tmp_tensor.scatter_(1, targets.view(-1, 1), 1)
                outputs = tmp_tensor
                batch_l0_grads = data - outputs
                batch_l1_grads = torch.zeros(self.batch_size, self.num_classes, l1.shape[1]).to(self.device)
                for i in range(self.batch_size):
                    for j in range(self.num_classes):
                        batch_l1_grads[i, j, :] = batch_l0_grads[i, j] * l1[i]
                l0_grads = torch.cat((l0_grads, batch_l0_grads), dim=0)
                l1_grads = torch.cat((l1_grads, batch_l1_grads), dim=0)
                cnt = cnt + 1
        torch.cuda.empty_cache()
        print("Per Element Gradient Computation is Completed")
        grads_list = list()
        for i in range(l0_grads.shape[0]):
            grads_list.append([l0_grads[i], l1_grads[i]])
        self.grads_per_elem = grads_list

    def _update_grads_val(self, theta_init, grads_currX=None, first_init=False):
        self.model.load_state_dict(theta_init)
        self.model.zero_grad()
        if first_init:
            with torch.no_grad():
                out, l1 = self.model(self.x_val)
                scores = F.softmax(out, dim=1)
                one_hot_label = torch.zeros(len(self.y_val), self.num_classes).to(self.device)
                one_hot_label.scatter_(1, self.y_val.view(-1, 1), 1)
                l0_grads = scores - one_hot_label
                l1_grads = torch.zeros(l1.shape[0], self.num_classes, l1.shape[1]).to(self.device)
                for i in range(l1.shape[0]):
                    for j in range(self.num_classes):
                        l1_grads[i, j, :] = l0_grads[i, j] * l1[i]
        # populate the gradients in model params based on loss.
        elif grads_currX is not None:
            # update params:
            with torch.no_grad():
                params = [param for param in self.model.parameters()]
                params[-1].data.sub_(self.eta * grads_currX[0])
                params[-2].data.sub_(self.eta * grads_currX[1])
                out, l1 = self.model(self.x_val)
                scores = F.softmax(out, dim=1)
                one_hot_label = torch.zeros(len(self.y_val), self.num_classes).to(self.device)
                one_hot_label.scatter_(1, self.y_val.view(-1, 1), 1)
                l0_grads = scores - one_hot_label
                l1_grads = torch.zeros(l1.shape[0], self.num_classes, l1.shape[1]).to(self.device)
                for i in range(l1.shape[0]):
                    for j in range(self.num_classes):
                        l1_grads[i, j, :] = l0_grads[i, j] * l1[i]
        self.grads_val_curr = torch.cat((torch.flatten(l0_grads.mean(dim=0)), torch.flatten(l1_grads.mean(dim=0))), dim=0)

    def eval_taylor_modular(self, grads):
        grads_val = self.grads_val_curr
        with torch.no_grad():
            grads_tensor = torch.cat(grads, dim=0)
            param_update = self.eta * grads_tensor
            gains = torch.matmul(param_update, grads_val)
        return gains

    # Updates gradients of set X + element (basically adding element to X)
    # Note that it modifies the inpute vector! Also grads_X is a list! grad_e is a tuple!
    def _update_gradients_subset(self, grads_X, element):
        grads_e = self.grads_per_elem[element]
        grads_X[0] += grads_e[0]
        grads_X[1] += grads_e[1]

    # Same as before i.e full batch case! No use of dataloaders here!
    # Everything is abstracted away in eval call
    def naive_greedy_max(self, budget, theta_init):
        start_time = time.time()
        self._compute_per_element_grads(theta_init)
        end_time = time.time()
        print("Per Element gradient computation time is: ", end_time - start_time)
        start_time = time.time()
        self._update_grads_val(theta_init, first_init=True)
        end_time = time.time()
        print("Updated validation set gradient computation time is: ", end_time - start_time)
        # Dont need the trainloader here!! Same as full batch version!
        numSelected = 0
        grads_currX = []  # basically stores grads_X for the current greedy set X
        greedySet = list()
        remainSet = list(range(self.N_trn))
        t_ng_start = time.time()  # naive greedy start time
        subset_size = int((len(self.grads_per_elem) / budget) * math.log(100))
        while (numSelected < budget):
            # Try Using a List comprehension here!
            t_one_elem = time.time()
            subset_selected = list(np.random.choice(np.array(list(remainSet)), size=subset_size, replace=False))
            rem_grads = [torch.cat((torch.flatten(self.grads_per_elem[x][0]), torch.flatten(self.grads_per_elem[x][1])), dim=0).reshape(1, -1) for x in subset_selected]
            gains = self.eval_taylor_modular(rem_grads)
            # Update the greedy set and remaining set
            bestId = subset_selected[torch.argmax(gains)]
            greedySet.append(bestId)
            remainSet.remove(bestId)
            # Update info in grads_currX using element=bestId
            if numSelected > 0:
                self._update_gradients_subset(grads_currX, bestId)
            else:  # If 1st selection, then just set it to bestId grads
                grads_currX = self.grads_per_elem[bestId]  # Making it a list so that is mutable!
            # Update the grads_val_current using current greedySet grads
            self._update_grads_val(theta_init, grads_currX)
            if numSelected % 1 == 0:
                # Printing bestGain and Selection time for 1 element.
                print("numSelected:", numSelected, "Time for 1:", time.time() - t_one_elem)
            numSelected += 1
        print("Naive greedy total time:", time.time() - t_ng_start)
        return list(greedySet), grads_currX


class WeightedSetFunctionLoader(object):
    def __init__(self, trainset, x_val, y_val, facloc_size, lam, model, loss_criterion,
                 loss_nored, eta, device, num_classes, batch_size):
        self.trainset = trainset  # assume its a sequential loader.
        self.x_val = x_val.to(device)
        self.y_val = y_val.to(device)
        self.facloc_size = facloc_size
        self.lam = lam
        self.model = model
        self.loss = loss_criterion  # Make sure it has reduction='none' instead of default
        self.loss_nored = loss_nored
        self.eta = eta  # step size for the one step gradient update
        # self.opt = optimizer
        self.device = device
        self.N_trn = len(trainset)
        self.grads_per_elem = None
        self.theta_init = None
        self.num_classes = num_classes
        self.batch_size = batch_size

    def _compute_per_element_grads(self, theta_init):
        self.model.load_state_dict(theta_init)
        batch_wise_indices = np.array(
            [list(BatchSampler(SequentialSampler(np.arange(self.N_trn)), self.batch_size, drop_last=False))][0])
        cnt = 0
        for batch_idx in batch_wise_indices:
            inputs = torch.cat(
                [self.trainset[x][0].view(-1, 1, self.trainset[x][0].shape[1], self.trainset[x][0].shape[2]) for x in
                 batch_idx], dim=0).type(torch.float)
            targets = torch.tensor([self.trainset[x][1] for x in batch_idx])
            inputs, targets = inputs.to(self.device), targets.to(self.device, non_blocking=True)
            if cnt == 0:
                with torch.no_grad():
                    out, l1 = self.model(inputs)
                    data = F.softmax(out, dim=1)
                tmp_tensor = torch.zeros(len(inputs), self.num_classes).to(self.device)
                tmp_tensor.scatter_(1, targets.view(-1, 1), 1)
                outputs = tmp_tensor
                l0_grads = data - outputs
                l1_grads = torch.zeros(self.batch_size, self.num_classes, l1.shape[1]).to(self.device)
                for i in range(self.batch_size):
                    for j in range(self.num_classes):
                        l1_grads[i, j, :] = l0_grads[i, j] * l1[i]
                cnt = cnt + 1
            else:
                with torch.no_grad():
                    out, l1 = self.model(inputs)
                    data = F.softmax(out, dim=1)
                tmp_tensor = torch.zeros(len(inputs), self.num_classes).to(self.device)
                tmp_tensor.scatter_(1, targets.view(-1, 1), 1)
                outputs = tmp_tensor
                batch_l0_grads = data - outputs
                batch_l1_grads = torch.zeros(self.batch_size, self.num_classes, l1.shape[1]).to(self.device)
                for i in range(self.batch_size):
                    for j in range(self.num_classes):
                        batch_l1_grads[i, j, :] = batch_l0_grads[i, j] * l1[i]
                l0_grads = torch.cat((l0_grads, batch_l0_grads), dim=0)
                l1_grads = torch.cat((l1_grads, batch_l1_grads), dim=0)
                cnt = cnt + 1
        torch.cuda.empty_cache()
        print("Per Element Gradient Computation is Completed")
        grads_list = list()
        for i in range(l0_grads.shape[0]):
            grads_list.append([l0_grads[i], l1_grads[i]])
        self.grads_per_elem = grads_list

    def _update_grads_val(self, theta_init, grads_currX=None, first_init=False):
        self.model.load_state_dict(theta_init)
        self.model.zero_grad()
        if first_init:
            with torch.no_grad():
                for i in range(10):
                    batch_out, batch_l1 = self.model(self.x_val[(i) * int((len(self.x_val)) / 10): (i + 1) * int((len(self.x_val)) / 10)])
                    batch_scores = F.softmax(batch_out, dim=1)
                    batch_one_hot_label = torch.zeros(
                        len(self.y_val[(i) * int(len(self.x_val) / 10): (i + 1) * int(len(self.x_val) / 10)]),
                        self.num_classes).to(self.device)
                    if (i == 0):
                        scores = batch_scores
                        l1 = batch_l1
                        one_hot_label = batch_one_hot_label.scatter_(1, self.y_val[
                                                                        (i) * int(len(self.x_val) / 10): (i + 1) * int(
                                                                            len(self.x_val) / 10)].view(-1, 1), 1)
                    else:
                        scores = torch.cat((scores, batch_scores), dim=0)
                        l1 = torch.cat((scores, batch_l1), dim=0)
                        one_hot_label = torch.cat((one_hot_label, batch_one_hot_label.scatter_(1, self.y_val[(i) * int(
                            len(self.x_val) / 10): (i + 1) * int(len(self.x_val) / 10)].view(-1, 1), 1)), dim=0)
                l0_grads = scores - one_hot_label
                l0_grads[0:int(self.facloc_size)] = self.lam * l0_grads[0:int(self.facloc_size)]
                l1_grads = torch.zeros(l0_grads.shape[0], self.num_classes, l1.shape[1]).to(self.device)
                for i in range(l0_grads.shape[0]):
                    for j in range(self.num_classes):
                        l1_grads[i, j, :] = l0_grads[i, j] * l1[i]
        # populate the gradients in model params based on loss.
        elif grads_currX is not None:
            # update params:
            with torch.no_grad():
                params = [param for param in self.model.parameters()]
                params[-1].data.sub_(self.eta * grads_currX[0])
                params[-2].data.sub_(self.eta * grads_currX[1])
                for i in range(10):
                    batch_out, batch_l1 = self.model(
                        self.x_val[(i) * int((len(self.x_val)) / 10): (i + 1) * int((len(self.x_val)) / 10)])
                    batch_scores = F.softmax(batch_out, dim=1)
                    batch_one_hot_label = torch.zeros(
                        len(self.y_val[(i) * int(len(self.x_val) / 10): (i + 1) * int(len(self.x_val) / 10)]),
                        self.num_classes).to(self.device)
                    if (i == 0):
                        scores = batch_scores
                        l1 = batch_l1
                        one_hot_label = batch_one_hot_label.scatter_(1, self.y_val[
                                                                        (i) * int(len(self.x_val) / 10): (i + 1) * int(
                                                                            len(self.x_val) / 10)].view(-1, 1), 1)
                    else:
                        scores = torch.cat((scores, batch_scores), dim=0)
                        l1 = torch.cat((scores, batch_l1), dim=0)
                        one_hot_label = torch.cat((one_hot_label, batch_one_hot_label.scatter_(1, self.y_val[(i) * int(
                            len(self.x_val) / 10): (i + 1) * int(len(self.x_val) / 10)].view(-1, 1), 1)), dim=0)
                l0_grads = scores - one_hot_label
                l0_grads[0:int(self.facloc_size)] = self.lam * l0_grads[0:int(self.facloc_size)]
                l1_grads = torch.zeros(l0_grads.shape[0], self.num_classes, l1.shape[1]).to(self.device)
                for i in range(l0_grads.shape[0]):
                    for j in range(self.num_classes):
                        l1_grads[i, j, :] = l0_grads[i, j] * l1[i]
        self.grads_val_curr = torch.cat((torch.flatten(l0_grads.mean(dim=0)), torch.flatten(l1_grads.mean(dim=0))), dim=0)

    def eval_taylor_modular(self, grads):
        grads_val = self.grads_val_curr
        with torch.no_grad():
            grads_tensor = torch.cat(grads, dim=0)
            param_update = self.eta * grads_tensor
            gains = torch.matmul(param_update, grads_val)
        return gains

    # Updates gradients of set X + element (basically adding element to X)
    # Note that it modifies the inpute vector! Also grads_X is a list! grad_e is a tuple!
    def _update_gradients_subset(self, grads_X, element):
        grads_e = self.grads_per_elem[element]
        grads_X[0] += grads_e[0]
        grads_X[1] += grads_e[1]

    # Same as before i.e full batch case! No use of dataloaders here!
    # Everything is abstracted away in eval call
    def naive_greedy_max(self, budget, theta_init):
        start_time = time.time()
        self._compute_per_element_grads(theta_init)
        end_time = time.time()
        print("Per Element gradient computation time is: ", end_time - start_time)
        start_time = time.time()
        self._update_grads_val(theta_init, first_init=True)
        end_time = time.time()
        print("Updated validation set gradient computation time is: ", end_time - start_time)
        # Dont need the trainloader here!! Same as full batch version!
        numSelected = 0
        grads_currX = []  # basically stores grads_X for the current greedy set X
        greedySet = list()
        remainSet = list(range(self.N_trn))
        t_ng_start = time.time()  # naive greedy start time
        subset_size = int((len(self.grads_per_elem) / budget) * math.log(100))
        while (numSelected < budget):
            # Try Using a List comprehension here!
            t_one_elem = time.time()
            subset_selected = list(np.random.choice(np.array(list(remainSet)), size=subset_size, replace=False))
            rem_grads = [torch.cat((torch.flatten(self.grads_per_elem[x][0]), torch.flatten(self.grads_per_elem[x][1])), dim=0).reshape(1, -1) for x in subset_selected]
            gains = self.eval_taylor_modular(rem_grads)
            # Update the greedy set and remaining set
            bestId = subset_selected[torch.argmax(gains)]
            greedySet.append(bestId)
            remainSet.remove(bestId)
            # Update info in grads_currX using element=bestId
            if numSelected > 0:
                self._update_gradients_subset(grads_currX, bestId)
            else:  # If 1st selection, then just set it to bestId grads
                grads_currX = self.grads_per_elem[bestId]  # Making it a list so that is mutable!
            # Update the grads_val_current using current greedySet grads
            self._update_grads_val(theta_init, grads_currX)
            if numSelected % 1000 == 0:
                # Printing bestGain and Selection time for 1 element.
                print("numSelected:", numSelected, "Time for 1:", time.time() - t_one_elem)
            numSelected += 1
        print("Naive greedy total time:", time.time() - t_ng_start)
        return list(greedySet), grads_currX


class SetFunctionSupervisedCRAIG(object):

    def __init__(self, trainset, trn_batch_size, if_convex, model, device):  # , valid_loader):

        self.trainset = trainset
        self.trn_batch_size = trn_batch_size
        self.if_convex = if_convex
        self.device = device  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model

    def distance(self, x, y, exp=2):

        n = x.size(0)
        m = y.size(0)
        d = x.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        dist = torch.pow(x - y, exp).sum(2)
        return dist  # torch.sqrt(dist)

        # dist = torch.abs(x-y).sum(2)
        # return dist

    def class_wise(self, bud):

        torch.cuda.empty_cache()

        classes = torch.unique(torch.tensor(self.trainset[:][1]))

        self.N = len(self.trainset)
        print(self.N)
        greedyList = []
        full_gamma = []

        for i in classes:

            idx = (self.y_trn == i).nonzero().flatten()
            idx.tolist()

            id_first = self.compute_score(idx)
            subset, gamma = self.lazy_greedy_max(math.ceil(bud * len(idx) / self.N), id_first)

            for j in range(len(subset)):
                greedyList.append(idx[subset[j]])
                full_gamma.append(gamma[j])

        c = list(zip(greedyList, full_gamma))
        random.shuffle(c)
        greedyList, full_gamma = zip(*c)

        return greedyList, full_gamma

    def compute_score(self, idxs):

        self.dist_mat = torch.zeros([self.curr_N, self.curr_N], dtype=torch.float32)

        actual_idxs = np.array(idxs)
        batch_wise_indices = actual_idxs[
            list(BatchSampler(SequentialSampler(actual_idxs), self.trn_batch_size, drop_last=False))]
        cnt = 0

        if self.if_convex:
            for batch_idx in batch_wise_indices:
                inputs = torch.cat(
                    [self.trainset[x][0].view(-1, 1, self.trainset[x][0].shape[1], self.trainset[x][0].shape[2]) for x
                     in batch_idx], dim=0).type(torch.float)
                if cnt == 0:
                    vector = inputs
                    cnt = cnt + 1
                else:
                    cnt = cnt + 1
                    vector = torch.cat((vector, inputs), dim=0)
        else:

            for batch_idx in batch_wise_indices:
                inputs = torch.cat(
                    [self.trainset[x][0].view(-1, 1, self.trainset[x][0].shape[1], self.trainset[x][0].shape[2]) for x
                     in batch_idx], dim=0).type(torch.float)
                targets = torch.tensor([self.trainset[x][1] for x in batch_idx])
                inputs, targets = inputs.to(self.device), targets.to(self.device, non_blocking=True)
                if cnt == 0:
                    with torch.no_grad():
                        data = F.softmax(self.model(inputs), dim=1)
                    tmp_tensor = torch.zeros(len(inputs), self.num_classes).to(self.device)
                    tmp_tensor.scatter_(1, targets.view(-1, 1), 1)
                    outputs = tmp_tensor
                    cnt = cnt + 1
                else:
                    cnt = cnt + 1
                    with torch.no_grad():
                        data = torch.cat((data, F.softmax(self.model(inputs), dim=1)), dim=0)
                    tmp_tensor = torch.zeros(len(inputs), self.num_classes).to(self.device)
                    tmp_tensor.scatter_(1, targets.view(-1, 1), 1)
                    outputs = torch.cat((outputs, tmp_tensor), dim=0)
            vector = data - outputs

        for i in range(math.ceil(len(vector) / self.trn_batch_size)):

            g_i = vector[i * self.trn_batch_size:(i + 1) * self.trn_batch_size]
            for j in range(math.ceil(len(vector) / self.trn_batch_size)):
                g_j = vector[j * self.trn_batch_size:(j + 1) * self.trn_batch_size]
                self.dist_mat[i * self.trn_batch_size: (i + 1) * self.trn_batch_size,
                j * self.trn_batch_size: (j + 1) * self.trn_batch_size] \
                    = self.distance(g_i, g_j)

        dist = self.dist_mat.sum(1)
        bestId = torch.argmin(dist).item()

        self.dist_mat = self.dist_mat.to(self.device)

        self.min_dist = self.dist_mat[bestId].to(self.device)

        return bestId

    def compute_gamma(self, idxs):

        gamma = [0 for i in range(len(idxs))]

        best = self.dist_mat[idxs]
        # print(best[0])
        rep = torch.argmin(best, axis=0)

        for i in rep:
            gamma[i] += 1

        return gamma

    def lazy_greedy_max(self, budget, id_first):

        self.gains = PriorityQueue()
        for i in range(self.curr_N):

            if i == id_first:
                continue
            curr_gain = (self.min_dist - torch.min(self.min_dist, self.dist_mat[i])).sum()
            self.gains.put((-curr_gain.item(), i))

        numSelected = 2
        second = self.gains.get()
        greedyList = [id_first, second[1]]
        self.min_dist = torch.min(self.min_dist, self.dist_mat[second[1]])

        while (numSelected < budget):

            # print(len(greedyList))
            if self.gains.empty():
                break

            elif self.gains.qsize() == 1:
                bestId = self.gains.get()[1]

            else:

                bestGain = -np.inf
                bestId = None

                while True:

                    first = self.gains.get()

                    if bestId == first[1]:
                        break

                    curr_gain = (self.min_dist - torch.min(self.min_dist, self.dist_mat[first[1]])).sum()
                    self.gains.put((-curr_gain.item(), first[1]))

                    if curr_gain.item() >= bestGain:
                        bestGain = curr_gain.item()
                        bestId = first[1]

            greedyList.append(bestId)
            numSelected += 1

            self.min_dist = torch.min(self.min_dist, self.dist_mat[bestId])

        gamma = self.compute_gamma(greedyList)
        return greedyList, gamma
