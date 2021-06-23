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
    async def federated_averaging(self, updates):
        """Aggregate weight updates from the clients using FedDA."""
        # intra-cluster model aggregation
        # inter-cluster model aggregation
        
        # Extract weights from the updates
        weights_received = self.extract_client_updates(updates)

        # Extract baseline model weights
        baseline_weights = self.algorithm.extract_weights()
        
        # Update server weights
        update = self.avg_att(baseline_weights, weights_received)

        return update


    """An iterative clustering strategy for BSs based on both the geo-locations and the traffic patterns."""
    def clustering(self, data, lng, lat, n_clusters=16, n_bs=100, pattern="tp"):
        df_ori = copy.deepcopy(data)
        # Initialization 
        df_ori['lng'] = lng
        df_ori['lat'] = lat
        df_ori['label'] = -1
        loc_init = np.zeros((n_clusters, 2)) # geo_location
        tp_init = np.zeros((n_clusters, df_ori.drop(['lng', 'lat', 'label'], axis=1).shape[1])) # traffic pattern
        geo_old_label = tp_old_label = [0] * n_bs
        
        # Iteration threshold
        iter_num = 20

        for i in range(iter_num):
            # print('{:}-th iter'.format(i))
            km_geo = cluster.KMeans(n_clusters=n_clusters, init=loc_init, n_init=1).fit(df_ori[['lng', 'lat']].values)
            km_tp = cluster.KMeans(n_clusters=n_clusters, init=tp_init, n_init=1).fit(
                df_ori.drop(['lng', 'lat', 'label'], axis=1).values
            )
            if pattern == 'geo':
                vm_geo = metrics.v_measure_score(geo_old_label, km_geo.labels_)
                # stable cluster center 
                if vm_geo == 1:
                    # print('Geolocation clustering converges at {:}-th iteration'.format(i+1))
                    break
                else:
                    # the traffic pattern information is considered by the geo-location clustering process
                    df_ori['label'] = km_tp.labels_
                    loc_init = df_ori.groupby(['label']).mean()[['lng', 'lat']].values
                    geo_old_label = km_geo.labels_
            elif pattern == 'tp':
                vm_tp = metrics.v_measure_score(tp_old_label, km_tp.labels_)
                # stable cluster center 
                if vm_tp == 1:
                    # print('Traffic pattern clustering converges at {:}'.format(i+1))
                    break
                else:
                    # the geo-location information is considered by the traffic pattern clustering process
                    df_ori['label'] = km_geo.labels_
                    tp_init = df_ori.groupby(['label']).mean().drop(['lng', 'lat'], axis=1).values
                    tp_old_label = km_tp.labels_
            else:
                print('wrong choice')
        if pattern == 'geo':
            return km_geo.labels_
        elif pattern == 'tp':
            return km_tp.labels_
        else:
            return km_tp.labels_
