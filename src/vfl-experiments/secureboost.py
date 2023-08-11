from datasets.utils import split_tabular_vertical
from datasets.datasets import Avazu, Income
from pipeline.backend.pipeline import PipeLine
from pipeline.component import Reader, DataTransform, Intersection, HeteroSecureBoost, Evaluation
from pipeline.interface import Data
import os

dataset = Income('../../data/income/', split='train')

###############################################
############### UPLOAD DATA ###################
##############################################
pipeline_upload = PipeLine().set_initiator(role='guest', party_id=9999).set_roles(guest=9999)
partition = 4

dense_data_guest = {"name": "income_guest", "namespace": f"experiment"}
dense_data_host = {"name": "income_host", "namespace": f"experiment"}

data_base = "../../data/"
pipeline_upload.add_upload_data(file=os.path.join(data_base, "income/train.csv"),
                                table_name=dense_data_guest["name"],             # table name
                                namespace=dense_data_guest["namespace"],         # namespace
                                head=1, partition=partition)               # data info

pipeline_upload.add_upload_data(file=os.path.join(data_base, "income/test.csv"),
                                table_name=dense_data_host["name"],
                                namespace=dense_data_host["namespace"],
                                head=1, partition=partition)


###############################################
############### TRAIN MODEL ###################
##############################################

pipeline = PipeLine() \
        .set_initiator(role='guest', party_id=9999) \
        .set_roles(guest=9999, host=10000)

reader_0 = Reader(name="reader_0")
# set guest parameter
reader_0.get_party_instance(role='guest', party_id=9999).component_param(
    table={"name": "income_guest", "namespace": "experiment"})
# set host parameter
reader_0.get_party_instance(role='host', party_id=10000).component_param(
    table={"name": "income_host", "namespace": "experiment"})

data_transform_0 = DataTransform(name="data_transform_0")
# set guest parameter
data_transform_0.get_party_instance(role='guest', party_id=9999).component_param(
    with_label=True)
data_transform_0.get_party_instance(role='host', party_id=[10000]).component_param(
    with_label=False)

intersect_0 = Intersection(name="intersect_0")

hetero_secureboost_0 = HeteroSecureBoost(name="hetero_secureboost_0",
                                         num_trees=5,
                                         bin_num=16,
                                         task_type="classification",
                                         objective_param={"objective": "cross_entropy"},
                                         encrypt_param={"method": "paillier"},
                                         tree_param={"max_depth": 3})

evaluation_0 = Evaluation(name="evaluation_0", eval_type="binary")


pipeline.add_component(reader_0)
pipeline.add_component(data_transform_0, data=Data(data=reader_0.output.data))
pipeline.add_component(intersect_0, data=Data(data=data_transform_0.output.data))
pipeline.add_component(hetero_secureboost_0, data=Data(train_data=intersect_0.output.data))
pipeline.add_component(evaluation_0, data=Data(data=hetero_secureboost_0.output.data))
pipeline.compile()

pipeline.fit()