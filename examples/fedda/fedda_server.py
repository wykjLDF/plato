"""
A federated learning server using FedDA.

Reference:

Zhang et al., "Dual Attention-Based Federated Learning for Wireless Traffic Prediction,"
in 2021 IEEE Conference on Computer Communications (INFOCOM).

https://chuanting.github.io/assets/pdf/ieee_infocom_2021.pdf
"""
from collections import OrderedDict

from plato.servers import fedavg
import fedda_model

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import copy
from sklearn import cluster, metrics


class Server(fedavg.Server):
    """A federated learning server using the FedDA algorithm."""
    def __init__(self):
        super().__init__()
        # TODO: Get clusters 
        # key - cluster id
        # value - ids of clients in that cluster
        # e.g., {'1': [1,2,4], '2': [3,5]}
        self.clusters = OrderedDict()
        # Get the quasi-global model
        self.quasi_global_weights = copy.deepcopy(self.get_quasi_global_model())
        # Initialize the global model as the quasi-global model
        self.algorithm.load_weights(self.quasi_global_weights)

    async def federated_averaging(self, updates):
        """Aggregate weight updates from the clients using FedDA."""
        # Extract weights from the updates
        weights_received = self.extract_client_updates(updates)
        
        # Extract global model weights
        global_weights = self.algorithm.extract_weights()

        # TODO: update local model weights per cluster
        local_weights = OrderedDict()

        # Perform intra-cluster model aggregation
        cluster_weights = OrderedDict()
        for cluster_id in self.clusters.keys():
            update = self.avg_att(local_weights[cluster_id], cluster_weights[cluster_id], self.quasi_global_weights)
            cluster_weights[cluster_id] 
     
        # Perform inter-cluster model aggregation
        update = self.avg_att(global_weights, self.cluster_weights)

        return update


    def clustering(self, client_updates):
        local_weights = OrderedDict()
        # TODO: cluster clients
        return local_weights
    
    def get_warm_data(self, data):
        # TODO: get warm data
        warm_xc, warm_xp, warm_y = None, None, None
        return  warm_xc, warm_xp, warm_y

    def get_quasi_global_model(self, data, warm_epochs=150):
        # sliding window parameters
        # closeness size: how many time slots before target are used to model closeness
        # period size: how many trend slots before target are used to model periodicity
        
        # Initialize the quasi-global model
        model = fedda_model.Model()

        # Randomly select clients
        
        # Get warm data to train the quasi-global model
        warm_xc, warm_xp, warm_y = self.get_warm_data(data)
        # TODO: concatenate
        warm_data = list(zip(*[warm_xc, warm_xp, warm_y]))
        warm_loader = DataLoader(warm_data, shuffle=False, batch_size=64)
        
        warm_criterion = torch.nn.MSELoss()
        warm_opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        warm_scheduler = torch.optim.lr_scheduler.MultiStepLR(warm_opt, milestones=[0.5*warm_epochs, 0.75*warm_epochs], gamma=0.1)

        for e in range(warm_epochs):
            warm_epoch_loss = []
            model.train()
            for batch_idx, (xc, xp, y) in enumerate(warm_loader):
                xc, xp, y = xc.float(), xp.float(), y.float()
                model.zero_grad()
                pred = model(xc, xp)
                loss = warm_criterion(y, pred)
                warm_epoch_loss.append(loss)
                loss.backward()
                warm_opt.step()

            warm_scheduler.step()

        return model.state_dict()

    def avg_dual_att(self, baseline_weights, weights_received, warm_weights, epsilon=1.0, dp=0.001, rho=0.2):
        """Perform dual attentive aggregation with the attention mechanism.
            epsilon: step size for aggregation
            dp: magnitude of normal noise in the randomization mechanism
            rho: importance of warm up model
        """
        att_update = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in weights_received[0].items()
        }

        atts = OrderedDict()
        warm_atts = OrderedDict()
        for name, weight in baseline_weights.items():
            atts[name] = self.trainer.zeros(len(weights_received))
            for i, update in enumerate(weights_received):
                delta = update[name]
                atts[name][i] = torch.linalg.norm(weight - delta)
            sw_diff = weight - warm_weights[name]
            warm_atts[name] = torch.FloatTensor(np.array(torch.linalg.norm(sw_diff)))
        
        # TODO: check format
        warm_tensor = torch.FloatTensor([v for k, v in warm_atts.items()])
        layer_w = F.softmax(warm_tensor, dim=0)
        
        for i, name in enumerate(baseline_weights.keys()):
            atts[name] = F.softmax(atts[name], dim=0)
            warm_att[name] = layer_w[i]

        for name, weight in baseline_weights.items():
            att_weight = self.trainer.zeros(weight.shape)
            for i, update in enumerate(weights_received):
                delta = update[name]
                att_weight += torch.mul(weight - delta, atts[name][i])

            att_weight += torch.mul(weight - warm_weights[name], rho*warm_atts[name])

            att_update[name] = -torch.mul(att_weight, epsilon)

        return att_update




