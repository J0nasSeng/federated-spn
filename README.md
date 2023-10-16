# Federated Circuits (FCs)
This repository contains the code of Federated Circuits and Federated Probabilistic Ciruits (FedPC).

## Abstract
Federated Learning (FL) trains machine learning models on distributed client data, eliminating the need for centralized data storage. It is typically categorized into horizontal (shared feature space, disjoint sample space), vertical (shared sample space, disjoint feature space), or hybrid FL (overlapping feature and sample spaces). However, existing methods only tackle individual FL scenarios separately and often suffer from a communication bottleneck, limiting real-world applicability. We propose Federated Circuits (FCs), a communication-efficient probabilistic approach that transforms FL into joint distribution learning by recursively partitioning client data to ensure invariance to data partitioning. We demonstrate FC's versatility in handling horizontal, vertical, and hybrid FL within a unified framework and that FCs efficiently scale Probabilistic Circuits (PCs) for large datasets like Imagenet.

## Getting Started
The repository is divided into several sub-directories: `parameter-server-spn` implements the classical horizontal FL scenario with the server computing a mixture of client-SPNs (TODO: support vertical scenario). The `network-aligned-spn` sub-directory contains the more general case in which we align the SPN structure and the communication network structure in order to perform hybrid FL (and horizontal/vertical as special cases). 

### parameter-server-spn
This sub-directory contains a `flwr` implementation of FedSPNs with depth $d=1$. To run this you have to install all packages listed in the `requirements.txt` in the root directory of this repository. Then, navigate to `src/parameter-server-spn` and run `python server.py` to start the server. If the server is running (might take a while due to possible data downloads), you can start the clients by invoking `python start_clients.py --clients N --gpus a1 a2 ...` where `N` refers to the number of clients being started and each `ai` refers to a GPU you want to select. If you run on CPU set `a = -1` and pass no further GPU-related argument. You can use the `config.py` file to configure the SPN architecture and other hyperparameters.

### network-aligned-spn
This sub-directory contains a `ray` implementation of FedSPNs which allows us to align the SPN structure and the network communication structure. Again, use the `config.py` to configure hyperparameters (e.g. SPN structure). To run FedSPNs invoke `python driver.py`. This script will start a `ray` driver process which emulates the communication network and it will start several clients which perform the training.

#### Hyperparameters for FedSPN
In the following we list the hyperparameters used for our experiments:

| Dataset       | Structure | Threshold | min_num_instances |
|---------------|-----------|-----------|-------------------|
| Credit        | learned   | 0.5       | 50                |
| Breast-Cancer | learned   | 0.4       | 300               |