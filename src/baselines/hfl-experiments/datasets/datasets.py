from torch.utils.data import Dataset, TensorDataset
from torch import FloatTensor, LongTensor
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd 
import torch
import os
import torch.nn.functional as F

class TabularDataset(Dataset):

    def __init__(self, x=None, y=None) -> None:
        super().__init__()
        self.targets: FloatTensor = y
        self.features: LongTensor = x

# TODO: rewrite 
class Avazu(TabularDataset):

    def __init__(self, path, split='train') -> None:
        super().__init__()
        self.split = split
        parse_date = lambda val : pd.datetime.strptime(val, '%y%m%d%H')
        self.target_encoder = TargetEncoder()
        self.sc = StandardScaler()
        self.train_data = pd.read_csv(os.path.join(path, 'train.csv'),
                                        parse_dates=['hour'], date_parser=parse_date)
        self.test_data = pd.read_csv(os.path.join(path, 'test.csv'),
                                        parse_dates=['hour'], date_parser=parse_date)
        if split == 'train' or split == 'valid':
            self._preprocess_train()
        elif split == 'test':
            self._preprocess_train()
            self._preprocess_test()
            
        else:
            raise ValueError(f'Split {split} not known')

    def _preprocess_train(self):
        self.train_data['month'] = self.train_data['hour'].dt.month
        self.train_data['dayofweek'] = self.train_data['hour'].dt.dayofweek
        self.train_data['day'] = self.train_data['hour'].dt.day
        self.train_data['hour_time'] = self.train_data['hour'].dt.hour

        # handle outliers
        col = ['C15', 'C16', 'C19', 'C21']
        for col in col:
            percentiles = self.train_data[col].quantile(0.98)
            if self.train_data[col].quantile(0.98) < 0.5 * self.train_data[col].max():
                self.train_data[col][self.train_data[col] >= percentiles] = percentiles

        self.train_data.drop(['id', 'hour'], axis = 1, inplace = True) 
        self.train_data.rename(columns={'click': 'y',
                   'hour_time': 'hour'},
          inplace=True, errors='raise')
        
        X = self.train_data.drop(['y'], axis=1)
        y = self.train_data['y']
        X = self.target_encoder.fit_transform(X, self.train_data['y'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state= 42)
        X_train = self.sc.fit_transform(X_train)
        X_test = self.sc.transform(X_test)

        if self.split == 'train':
            self.features = torch.from_numpy(X_train)
            self.targets = F.one_hot(torch.from_numpy(y_train)).to(torch.float32)
            self.dataset = TensorDataset(self.features, self.targets)
        elif self.split == 'valid':
            self.features = torch.from_numpy(X_test)
            self.targets = F.one_hot(torch.from_numpy(y_test)).to(torch.float32)
            self.dataset = TensorDataset(self.features, self.targets)

    def _preprocess_test(self):
        self.test_data['month'] = self.test_data['hour'].dt.month
        self.test_data['dayofweek'] = self.test_data['hour'].dt.dayofweek
        self.test_data['day'] = self.test_data['hour'].dt.day
        self.test_data['hour_time'] = self.test_data['hour'].dt.hour

        # handle outliers
        col = ['C15', 'C16', 'C19', 'C21']
        for col in col:
            percentiles = self.test_data[col].quantile(0.98)
            if self.test_data[col].quantile(0.98) < 0.5 * self.test_data[col].max():
                self.test_data[col][self.test_data[col] >= percentiles] = percentiles

        self.test_data.drop(['id', 'hour'], axis = 1, inplace = True) 
        self.test_data.rename(columns={'click': 'y',
                   'hour_time': 'hour'},
          inplace=True, errors='raise')
        
        X = self.test_data.drop(['y'], axis=1)
        y = self.test_data['y']
        X = self.target_encoder.transform(X, self.test_data['y'])
        X_test = self.sc.transform(X)
        y = F.one_hot(torch.from_numpy(y)).to(torch.float32)
        self.dataset = TensorDataset(torch.from_numpy(X_test), y)
        

class Income(TabularDataset):

    def __init__(self, path, split='train') -> None:
        super().__init__()

        self.split = split
        self.target_encoder = TargetEncoder()
        self.sc = StandardScaler()
        self.train_data = pd.read_csv(os.path.join(path, 'train.csv'))
        self.test_data = pd.read_csv(os.path.join(path, 'test.csv'))
        self._preprocess()
        
    def _preprocess(self):
        self.train_data.rename(columns={'income_>50K': 'income'}, inplace=True)

        # handle outliers
        col = ['capital-loss', 'capital-gain', 'fnlwgt']
        for col in col:
            percentiles = self.train_data[col].quantile(0.98)
            if self.train_data[col].quantile(0.98) < 0.5 * self.train_data[col].max():
                self.train_data[col][self.train_data[col] >= percentiles] = percentiles

        X = self.train_data.drop(['income'], axis=1)
        y = self.train_data['income']
        X = self.target_encoder.fit_transform(X, self.train_data['income'])
        X_train, X_tv, y_train, y_tv = train_test_split(X, y, stratify=y, test_size=0.3, random_state= 42)
        X_train = self.sc.fit_transform(X_train)
        X_tv = self.sc.transform(X_tv)

        X_valid, X_test, y_valid, y_test = train_test_split(X_tv, y_tv, stratify=y_tv, test_size=0.5, random_state=42)

        if self.split == 'train':
            self.features = torch.from_numpy(X_train).to(torch.float32)
            self.targets = F.one_hot(torch.from_numpy(y_train.to_numpy())).to(torch.float32)
            self.dataset = TensorDataset(self.features, self.targets)
        elif self.split == 'valid':
            self.features = torch.from_numpy(X_valid).to(torch.float32)
            self.targets = F.one_hot(torch.from_numpy(y_valid.to_numpy())).to(torch.float32)
            self.dataset = TensorDataset(self.features, self.targets)
        elif self.split == 'test':
            self.features = torch.from_numpy(X_test).to(torch.float32)
            self.targets = F.one_hot(torch.from_numpy(y_test.to_numpy())).to(torch.float32)
            self.dataset = TensorDataset(self.features, self.targets)