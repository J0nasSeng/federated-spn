"""
    This file starts the driver process of a ray cluster.
    It then coordinates training of the clients. 
    Note that this just simulates the ring-reduce network topology
    proposed in our FedSPN paper.
    Nevertheless, semantically it's performing the same operations
    as ring reduce algorithm.
"""

import ray

ray.init()