"""
A federated learning server using FedDA.

Reference:

Zhang et al., "Dual Attention-Based Federated Learning for Wireless Traffic Prediction,"
in 2021 IEEE Conference on Computer Communications (INFOCOM).

https://chuanting.github.io/assets/pdf/ieee_infocom_2021.pdf
"""
from collections import OrderedDict

from plato.servers import fedavg
from examples.fedatt import fedatt_server

import torch
import torch.nn.functional as F
import numpy as np
import copy
from sklearn import cluster, metrics


class Server(fedatt_server.Server):
    """A federated learning server using the FedDA algorithm."""
    def __init__(self):
        super().__init__()
        self.cluster_weights = OrderedDict()

    async def federated_averaging(self, updates):
        """Aggregate weight updates from the clients using FedDA."""
        # Extract weights from the updates
        weights_received = self.extract_client_updates(updates)
        
        # Extract server model weights
        server_weights = self.algorithm.extract_weights()

        # Perform intra-cluster model aggregation
        local_weights = self.clustering(weights_received)
        for cluster_id in local_weights.keys():
            self.cluster_weights[cluster_id] = self.avg_att(local_weights[cluster_id], self.cluster_weights[cluster_id])
     
        # Perform inter-cluster model aggregation
        update = self.avg_att(server_weights, self.cluster_weights)

        return update


    def clustering(self, client_updates):
        local_weights = OrderedDict()
        # TODO: cluster clients
        return local_weights
