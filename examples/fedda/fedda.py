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


def main():
    """ A Plato federated learning training session using the FedDA algorithm. """
    server = fedda_server.Server()

    server.run()


if __name__ == "__main__":
    main()
