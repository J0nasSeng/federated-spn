# Federated SPN (FedSPN)
This repository contains the code of Federated Sum-Product Networks (FedSPN).

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