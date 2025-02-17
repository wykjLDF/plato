
## Design

The Plato framework is designed to be extensible, hopefully making it easy to add new data sources for datasets, models, and custom trainers for models. This document discusses the current design of the framework from a software engineering perspective.

This framework makes extensive use of object oriented subclassing with the help of Python 3's [ABC library](https://docs.python.org/3/library/abc.html). It is a good idea to review Python 3's support for base classes with abstract methods before proceeding. It also makes sporadic use of Python 3's [Data Classes](https://docs.python.org/3/library/dataclasses.html).

### Configuration parameters

All configuration parameters are globally accessed using the Singleton `Config` class globally (found in `config.py`). They are read from a configuration file when the clients and the servers launch, and the configuration file follows the YAML format for the sake of simplicity and readability. These parameters include parameters specific to the dataset, data distribution, trainer, the federated learning algorithm, server configuration, and cross-silo training.

Either a command-line argument (`-c` or `--config`) or an environment variable `config_file` can be used to specify the location of the configuration file. Use `Config()` anywhere in the framework to access these configuration parameters.

### Extensible modules

This framework breaks commonly shared components in a federated learning training workload into extensible modules that are as independent as possible.

#### Data sources

A `Datasource` instance is used to obtain the dataset, labels, and any data augmentation. For example, the PyTorch `DataLoader` class in `torch.utils.data` can be used to load the MNIST dataset; `Datasets` classes in the `HuggingFace` framework can also be used as a data source to load datasets.

A data source must subclass the `Datasource` abstract base classes in `datasources/base.py`. This class may use third-party frameworks to load datasets, and may add additional functionality to support build-in transformations.

The external interface of this module is contained in `datasources/registry.py`. The registry contains a list of all existing datasources in the framework, so that they can be discovered and loaded. Its most important function is `get()`, which returns a `DataSource` instance.

#### Samplers 

A `Sampler` is responsible for sampling a dataset for local training or testing at each client in the federated learning workload. This is used to *simulate* a local dataset that is available locally at the client, using either an i.i.d. or non-i.i.d. distribution. For non-i.i.d. distributions, an example sampler that is based on the Dirichlet distribution (with a configurable concentration bias) is provided. Samplers are passed as one of the parameters to a PyTorch `Dataloader` or MindSpore `Dataset` instance.

#### Models

Each model is created by subclassing the `Model` abstract base class in `models/base.py`. This base class is a valid PyTorch `nn.Module` with several additional abstract methods that support other functionality throughout the framework. In particular, any subclass must have static methods to determine whether a string model name (e.g., `cifar_resnet_18`) is valid and to create a model object from a string name, a number of outputs, and an initializer.

The external interface of this module is contained in `models/registry.py`. Just like `datasets/registry.py`, there is a list of all existing models in the framework so that they can be discovered and loaded. The registry similarly contains a `get()` function that returns the corresponding `Model` as specified. 

### Federated learning algorithms

Most federated learning algorithms can be divided into three components: a *client*, a *server*, and a *algorithm*. The *client* implements all algorithm logic on the client side, while the *server* implements all algorithm logic on the server side. Both the client and the server should also be neutral across various deep learning frameworks. All algorithm logic that is framework-specific should be included in an *algorithm* module, found in `algorithms/`.
