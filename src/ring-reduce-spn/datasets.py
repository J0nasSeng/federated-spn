from torch.utils.data import TensorDataset
import numpy as np
import os
import arff
from scipy.io.arff import loadarff
import pandas as pd
import xml.etree.ElementTree as ET

def one_hot(y, values):
    if len(y.shape) != 1:
        return y
    values = np.array(sorted(list(set(values))))
    return np.array([values == v for v in y], dtype=np.int8)


def transpose_list(data):
    return list(map(list, zip(*data)))

def preproc_arff_data(raw_data, labels):
    data = raw_data["data"]
    data_transposed = transpose_list(data)
    labels = [child.attrib["name"] for child in labels.getroot()]
    labels_idx = np.asarray([elem[0] in labels for elem in raw_data["attributes"]])
    numeric_idx = np.asarray([elem[1] == "NUMERIC" for elem in raw_data["attributes"]])
    values = [elem[1] for elem in raw_data["attributes"]]  # the range of ohe

    num_data_rows = len(data)
    num_labels = len(labels)
    num_data_cols = len(raw_data["attributes"])
    num_input_cols = num_data_cols - num_labels

    # split input and labels
    input_transposed = np.asarray([input for i, input in enumerate(data_transposed) if labels_idx[i] == False])
    values_input = [value for i, value in enumerate(values) if labels_idx[i] == False]
    labels = [
        one_hot(np.asarray(label), values[i]) for i, label in enumerate(data_transposed) if labels_idx[i] == True
    ]  # do we need to ohe labels?
    labels_ohe = np.swapaxes(np.asarray(labels), 0, 1).reshape(
        num_data_rows, -1
    )  # shape is now (#instance, #labels, #ohe)

    ohe_data_arr = None
    for i in range(num_input_cols):
        if ohe_data_arr is None:
            if numeric_idx[i] == False:
                ohe_data_arr = one_hot(input_transposed[i], values[i]).reshape(-1, num_data_rows)
            else:
                ohe_data_arr = input_transposed[i].reshape(-1, num_data_rows)
        else:
            if numeric_idx[i] == False:
                ohe_data_arr = np.concatenate(
                    (ohe_data_arr, one_hot(input_transposed[i], values[i]).reshape(-1, num_data_rows)), axis=0
                )
            else:
                ohe_data_arr = np.concatenate((ohe_data_arr, input_transposed[i].reshape(-1, num_data_rows)), axis=0)

    return transpose_list(ohe_data_arr), labels_ohe

def get_categorical_data(path, name):

    train = arff.load(open(path + "/categorical/" + name + "/" + name + "-train.arff", "r"))
    test = arff.load(open(path + "/categorical/" + name + "/" + name + "-test.arff", "r"))
    labels = ET.parse(path + "/categorical/" + name + "/" + name + ".xml")

    train_input, train_labels = preproc_arff_data(train, labels)
    test_input, test_labels = preproc_arff_data(test, labels)

    return (train_input, train_labels, test_input, test_labels)

def get_medical_data():
    train_x, train_y, test_x, test_y = get_categorical_data('../../datasets/', 'medical')
    train_joint, test_joint = np.hstack((train_x, train_y)), np.hstack((test_x, test_y))
    train_ds, test_ds = TensorDataset(train_joint), TensorDataset(test_joint)
    return train_ds, test_ds