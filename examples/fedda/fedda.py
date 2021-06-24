"""
A federated learning training session using FedDA.

Reference:

Zhang et al., "Dual Attention-Based Federated Learning for Wireless Traffic Prediction,"
in 2021 IEEE Conference on Computer Communications (INFOCOM).

https://chuanting.github.io/assets/pdf/ieee_infocom_2021.pdf
"""
import os

os.environ['config_file'] = 'examples/fedda/fedda_MNIST_lenet5.yml'

import fedda_server
import fedda_model
from plato.trainers import basic
from plato.clients import simple


def main():
    """ A Plato federated learning training session using the FedDA algorithm. """
    model = fedda_model.Model()
    trainer = basic.Trainer(model=model)
    client = simple.Client(trainer=trainer)
    server = fedda_server.Server(trainer=trainer)

    server.run()
    

if __name__ == "__main__":
    main()
