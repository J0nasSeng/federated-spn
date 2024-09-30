from fedlab.utils.dataset import BasicPartitioner

class IncomePartitioner(BasicPartitioner):

    num_classes = 2
    num_features = 14

class AvazuPartitioner(BasicPartitioner):

    num_classes = 2
    num_features = 16

class BreasCancerPartitioner(BasicPartitioner):

    num_classes = 2
    num_features = 31

class GimmeCreditPartitioner(BasicPartitioner):

    num_classes = 2
    num_features = 11

class BAFPartitioner(BasicPartitioner):

    num_classes = 2
    num_features = 30

class SantanderPartitioner(BasicPartitioner):

    num_classes = 2
    num_features = 200

class PartitionerFactory:

    def __init__(self):
        self.partitioners = {
            'income': IncomePartitioner,
            'avazu': AvazuPartitioner,
            'breast-cancer': BreasCancerPartitioner,
            'credit': GimmeCreditPartitioner,
            'baf': BAFPartitioner,
            'santander': SantanderPartitioner
        }

    def get_partitioner_cls(self, dataset):
        return self.partitioners[dataset.name]