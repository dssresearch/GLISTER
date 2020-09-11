import numpy as np
import time
import torch


## One Step Set Functions on Validation Loss
## Batch and Pytorch Dataloader variants implemented.


class SetFunctionBatch(object):

    def __init__(self, X_trn, Y_trn, X_val, Y_val, model, params_init, loss_criterion, eta):
        self.x_trn = X_trn
        self.x_val = X_val
        self.y_trn = Y_trn
        self.y_val = Y_val
        self.model = model
        self.loss = loss_criterion  # Make sure it has reduction='none' instead of default
        self.eta = eta  # step size for the one step gradient update
        # self.device = device
        self.theta_init = params_init
        # self.theta_init = copy.deepcopy(model.state_dict())
        self.N_trn = X_trn.shape[0]
        self.grads_per_elem = None
        self.grads_val_curr = None

    # Update the grads_val based on given grads_currX or first init
    def _update_grads_val(self, grads_currX=None, first_init=False):
        self.model.load_state_dict(self.theta_init)
        self.model.zero_grad()
        if first_init:
            scores = self.model(self.x_val)
            loss = -1.0 * self.loss(scores, self.y_val).mean()
            loss.backward()  # populate the gradients in model params based on loss.
        elif grads_currX is not None:
            # update params:
            with torch.no_grad():
                for i, param in enumerate(self.model.parameters()):
                    param.data.sub_(self.eta * grads_currX[i])
            scores = self.model(self.x_val)
            loss = -1.0 * self.loss(scores, self.y_val).mean()
            loss.backward()  # populate the gradients in model params based on loss.
        # populate the grads_val list
        self.grads_val_curr = [param.grad.data for i, param in enumerate(self.model.parameters())]
        self.model.zero_grad()  # reset parm.grads to zero!

    def _compute_per_element_grads(self):
        self.model.load_state_dict(self.theta_init)
        self.model.zero_grad()
        scores = self.model(self.x_trn)
        losses = self.loss(scores, self.y_trn)
        N = self.N_trn
        grads_vec = [0 for _ in range(N)]  # zero is just a placeholder
        for item in range(N):
            grads_vec[item] = torch.autograd.grad(losses[item], self.model.parameters(), retain_graph=True)
        self.grads_per_elem = grads_vec

    # computes the gain for elem
    def eval_taylor(self, grads_elem):
        grads_val = self.grads_val_curr
        dot_prod = 0
        self.model.load_state_dict(self.theta_init)
        with torch.no_grad():
            for i, param in enumerate(self.model.parameters()):
                dot_prod += torch.sum(grads_val[i] * (param.data - self.eta * grads_elem[i]))
        return dot_prod.data

    # Updates gradients of set X + element (basically adding element to X)
    # Note that it modifies the inpute vector! Also grads_X is a list! grad_e is a tuple!
    # Hence can modify grads_X in place!
    def _update_gradients_subset(self, grads_X, element):
        grads_e = self.grads_per_elem[element]
        for i, _ in enumerate(self.model.parameters()):
            grads_X[i] += grads_e[i]

    def naive_greedy_max(self, budget):
        self._compute_per_element_grads()
        self._update_grads_val(first_init=True)
        print("Computed Train set and Val gradients")
        numSelected = 0
        grads_currX = []  # basically stores grads_X for the current greedy set X
        greedySet = set()
        remainSet = set(range(self.N_trn))
        t_ng_start = time.time()  # naive greedy start time
        while (numSelected < budget):
            # Try Using a List comprehension here!
            bestGain = -np.inf  # value for current iteration (validation loss)
            bestId = -1  # element to pick
            t_one_elem = time.time()

            for i in remainSet:
                grads_i = self.grads_per_elem[i]
                val_i = self.eval_taylor(grads_i)
                if val_i > bestGain:
                    bestGain = val_i
                    bestId = i

            # Update the greedy set and remaining set
            greedySet.add(bestId)
            remainSet.remove(bestId)
            # Update info in grads_currX using element=bestId
            if numSelected > 0:
                self._update_gradients_subset(grads_currX, bestId)
            else:  # If 1st selection, then just set it to bestId grads
                grads_currX = list(
                    self.grads_per_elem[bestId])  # Making it a list so that is mutable!
            # Update the grads_val_current using current greedySet grads
            self._update_grads_val(grads_currX)

            if numSelected % 500 == 0:
                # Printing bestGain and Selection time for 1 element.
                print("numSelected:", numSelected, "Time for 1:", time.time() - t_one_elem, "bestGain:", bestGain)
            numSelected += 1
        print("Naive greedy total time:", time.time() - t_ng_start)
        return list(greedySet), grads_currX


class SetFunctionLoader_2(object):

    def __init__(self, trainloader, validset, model, loss_criterion,
                 loss_nored, eta, device):
        self.trainloader = trainloader  # assume its a sequential loader.
        # self.valloader = valloader      # assume its a sequential loader.
        self.x_val = validset.dataset.data[validset.indices].to(device, dtype=torch.float).view(len(validset.indices),
                                                                                                1, 28, 28)
        self.y_val = validset.dataset.targets[validset.indices].to(device)
        self.model = model
        self.loss = loss_criterion  # Make sure it has reduction='none' instead of default
        self.loss_nored = loss_nored
        self.eta = eta  # step size for the one step gradient update
        # self.opt = optimizer
        self.device = device
        self.N_trn = len(trainloader.sampler)
        self.grads_per_elem = None
        self.theta_init = None

    def _compute_per_element_grads(self, theta_init):
        grads_vec = [0 for i in range(self.N_trn)]
        counter = 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            # print(batch_idx)
            inputs, targets = inputs.to(self.device), targets.to(self.device, non_blocking=True)
            self.model.load_state_dict(theta_init)
            scores = self.model(inputs)
            losses = self.loss_nored(scores, targets)
            for i, loss_i in enumerate(losses):
                self.model.zero_grad()
                grads_vec[i + counter] = [torch.autograd.grad(loss_i, self.model.parameters(), retain_graph=True)[-1]]
            params = [param for param in self.model.parameters()]
            for param in params:
                param.grad = None
                param.detach()
            torch.cuda.empty_cache()
            counter += len(targets)
            # if (batch_idx % 100 == 0):
            # print(torch.cuda.memory_allocated())
        print("Per Element Gradient Computation is Completed")
        self.grads_per_elem = grads_vec

    def _update_grads_val(self, theta_init, grads_currX=None, first_init=False):
        self.model.load_state_dict(theta_init)
        self.model.zero_grad()

        if first_init:
            scores = self.model(self.x_val)
            loss = -1.0 * self.loss(scores, self.y_val).mean()
            #print(torch.cuda.memory_allocated())
            self.grads_val_curr = [torch.autograd.grad(loss, self.model.parameters(), retain_graph=False)[-1]]
        # populate the gradients in model params based on loss.
        elif grads_currX is not None:
            # update params:
            with torch.no_grad():
                params = [param for param in self.model.parameters()]
                params[-1].data.sub_(self.eta * grads_currX[0])
            scores = self.model(self.x_val)
            loss = -1.0 * self.loss(scores, self.y_val).mean()
            self.grads_val_curr = [torch.autograd.grad(loss, self.model.parameters(), retain_graph=False)[-1]]
        self.model.zero_grad()  # reset parm.grads to zero!

    def eval_taylor(self, grads_elem, theta_init):
        grads_val = self.grads_val_curr
        dot_prod = 0
        self.model.load_state_dict(theta_init)
        with torch.no_grad():
            params = [param for param in self.model.parameters()]
            dot_prod += torch.sum(grads_val[0] * (params[-1].data - self.eta * grads_elem[0]))
        return dot_prod.data

    def eval_taylor_modular(self, grads, theta_init):
        grads_val = self.grads_val_curr
        self.model.load_state_dict(theta_init)
        params = [param for param in self.model.parameters()]
        temp_tensor = params[-1].data.detach()
        grads_tensor = torch.cat(grads, dim=0)
        param_update = temp_tensor - self.eta * grads_tensor
        gains = torch.matmul(param_update, grads_val[0])
        return gains

    # Updates gradients of set X + element (basically adding element to X)
    # Note that it modifies the inpute vector! Also grads_X is a list! grad_e is a tuple!
    def _update_gradients_subset(self, grads_X, element):
        grads_e = self.grads_per_elem[element]
        grads_X[0] += grads_e[0]

    # Same as before i.e full batch case! No use of dataloaders here!
    # Everything is abstracted away in eval call
    def naive_greedy_max(self, budget, theta_init):
        self._compute_per_element_grads(theta_init)
        self._update_grads_val(theta_init, first_init=True)
        print("Computed Train set and Val gradients")
        # Dont need the trainloader here!! Same as full batch version!
        numSelected = 0
        grads_currX = []  # basically stores grads_X for the current greedy set X
        greedySet = set()
        remainSet = set(range(self.N_trn))
        t_ng_start = time.time()  # naive greedy start time
        while (numSelected < budget):
            # Try Using a List comprehension here!
            t_one_elem = time.time()
            rem_grads = [self.grads_per_elem[x][0].view(1, self.grads_per_elem[0][0].shape[0]) for x in remainSet]
            gains = self.eval_taylor_modular(rem_grads, theta_init)
            # Update the greedy set and remaining set
            bestId = list(remainSet)[torch.argmax(gains)]
            greedySet.add(bestId)
            remainSet.remove(bestId)
            # Update info in grads_currX using element=bestId
            if numSelected > 0:
                self._update_gradients_subset(grads_currX, bestId)
            else:  # If 1st selection, then just set it to bestId grads
                grads_currX = list(
                    self.grads_per_elem[bestId])  # Making it a list so that is mutable!
            # Update the grads_val_current using current greedySet grads
            self._update_grads_val(theta_init, grads_currX)

            if numSelected % 1000 == 0:
                # Printing bestGain and Selection time for 1 element.
                print("numSelected:", numSelected, "Time for 1:", time.time() - t_one_elem)
            numSelected += 1
        print("Naive greedy total time:", time.time() - t_ng_start)
        return list(greedySet), grads_currX


class WeightedSetFunctionLoader(object):

    def __init__(self, trainloader, validset, facloc_size, lam, model, loss_criterion,
                 loss_nored, eta, device):
        self.trainloader = trainloader  # assume its a sequential loader.
        # self.valloader = valloader      # assume its a sequential loader.
        self.x_val = validset.dataset.data[validset.indices].to(device, dtype=torch.float).view(len(validset.indices),
                                                                                                1, 28, 28)
        self.y_val = validset.dataset.targets[validset.indices].to(device)
        self.facloc_size = facloc_size
        self.lam = lam
        self.model = model
        self.loss = loss_criterion  # Make sure it has reduction='none' instead of default
        self.loss_nored = loss_nored
        self.eta = eta  # step size for the one step gradient update
        # self.opt = optimizer
        self.device = device
        self.N_trn = len(trainloader.sampler)
        self.grads_per_elem = None
        self.theta_init = None

    def _compute_per_element_grads(self, theta_init):
        grads_vec = [0 for i in range(self.N_trn)]
        counter = 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            # print(batch_idx)
            inputs, targets = inputs.to(self.device), targets.to(self.device, non_blocking=True)
            self.model.load_state_dict(theta_init)
            scores = self.model(inputs)
            losses = self.loss_nored(scores, targets)
            for i, loss_i in enumerate(losses):
                self.model.zero_grad()
                grads_vec[i + counter] = [torch.autograd.grad(loss_i, self.model.parameters(), retain_graph=True)[-1]]
            params = [param for param in self.model.parameters()]
            for param in params:
                param.grad = None
                param.detach()
            torch.cuda.empty_cache()
            counter += len(targets)
            # if (batch_idx % 100 == 0):
            # print(torch.cuda.memory_allocated())
        print("Per Element Gradient Computation is Completed")
        self.grads_per_elem = grads_vec

    def _update_grads_val(self, theta_init, grads_currX=None, first_init=False):
        self.model.load_state_dict(theta_init)
        self.model.zero_grad()
        if first_init:
            #print(torch.cuda.memory_allocated())
            facloc_tensor = 0
            for i in range(10):
                scores = self.model(self.x_val[(i) * int((self.facloc_size)/10) : (i+1) * int((self.facloc_size)/10)])
                loss = self.lam * self.loss(scores, self.y_val[(i) * int((self.facloc_size)/10):(i+1) *int((self.facloc_size)/10)]).mean()
                facloc_tensor += torch.autograd.grad(loss, self.model.parameters(), retain_graph=False)[-1]
                self.model.zero_grad()
            scores = self.model(self.x_val[self.facloc_size:])
            loss = self.lam * self.loss(scores, self.y_val[self.facloc_size:]).mean()
            val_tensor = torch.autograd.grad(loss, self.model.parameters(), retain_graph=False)[-1]
            self.grads_val_curr = [facloc_tensor + val_tensor]
        # populate the gradients in model params based on loss.
        elif grads_currX is not None:
            # update params:
            with torch.no_grad():
                params = [param for param in self.model.parameters()]
                params[-1].data.sub_(self.eta * grads_currX[0])
            facloc_tensor = 0
            for i in range(10):
                scores = self.model(self.x_val[(i) * int((self.facloc_size) / 10) : (i+1) * int((self.facloc_size) / 10)])
                loss = self.lam * self.loss(scores, self.y_val[(i) * int((self.facloc_size) / 10):
                                                               (i+1) * int((self.facloc_size) / 10)]).mean()
                facloc_tensor += torch.autograd.grad(loss, self.model.parameters(), retain_graph=False)[-1]
                self.model.zero_grad()
            scores = self.model(self.x_val[self.facloc_size:])
            loss = self.lam * self.loss(scores, self.y_val[self.facloc_size:])
            val_tensor = torch.autograd.grad(loss, self.model.parameters(), retain_graph=False)[-1]
            self.grads_val_curr = [facloc_tensor + val_tensor]
        self.model.zero_grad()  # reset parm.grads to zero!

    def eval_taylor(self, grads_elem, theta_init):
        grads_val = self.grads_val_curr
        dot_prod = 0
        self.model.load_state_dict(theta_init)
        with torch.no_grad():
            params = [param for param in self.model.parameters()]
            dot_prod += torch.sum(grads_val[0] * (params[-1].data - self.eta * grads_elem[0]))
        return dot_prod.data

    def eval_taylor_modular(self, grads, theta_init):
        grads_val = self.grads_val_curr
        self.model.load_state_dict(theta_init)
        params = [param for param in self.model.parameters()]
        temp_tensor = params[-1].data.detach()
        grads_tensor = torch.cat(grads, dim=0)
        param_update = temp_tensor - self.eta * grads_tensor
        gains = torch.matmul(param_update, grads_val[0])
        return gains

    # Updates gradients of set X + element (basically adding element to X)
    # Note that it modifies the inpute vector! Also grads_X is a list! grad_e is a tuple!
    def _update_gradients_subset(self, grads_X, element):
        grads_e = self.grads_per_elem[element]
        grads_X[0] += grads_e[0]

    # Same as before i.e full batch case! No use of dataloaders here!
    # Everything is abstracted away in eval call
    def naive_greedy_max(self, budget, theta_init):
        self._compute_per_element_grads(theta_init)
        self._update_grads_val(theta_init, first_init=True)
        print("Computed Train set and Val gradients")
        # Dont need the trainloader here!! Same as full batch version!
        numSelected = 0
        grads_currX = []  # basically stores grads_X for the current greedy set X
        greedySet = set()
        remainSet = set(range(self.N_trn))
        t_ng_start = time.time()  # naive greedy start time
        while (numSelected < budget):
            # Try Using a List comprehension here!
            t_one_elem = time.time()
            rem_grads = [self.grads_per_elem[x][0].view(1, self.grads_per_elem[0][0].shape[0]) for x in remainSet]
            gains = self.eval_taylor_modular(rem_grads, theta_init)
            # Update the greedy set and remaining set
            bestId = list(remainSet)[torch.argmax(gains)]
            greedySet.add(bestId)
            remainSet.remove(bestId)
            # Update info in grads_currX using element=bestId
            if numSelected > 0:
                self._update_gradients_subset(grads_currX, bestId)
            else:  # If 1st selection, then just set it to bestId grads
                grads_currX = list(
                    self.grads_per_elem[bestId])  # Making it a list so that is mutable!
            # Update the grads_val_current using current greedySet grads
            self._update_grads_val(theta_init, grads_currX)

            if numSelected % 1000 == 0:
                # Printing bestGain and Selection time for 1 element.
                print("numSelected:", numSelected, "Time for 1:", time.time() - t_one_elem)
            numSelected += 1
        print("Naive greedy total time:", time.time() - t_ng_start)
        return list(greedySet), grads_currX
