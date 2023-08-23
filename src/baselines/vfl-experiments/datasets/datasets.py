from torch.utils.data import Dataset
from torch import FloatTensor, LongTensor
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd 
import torch
import os

class TabularDataset(Dataset):

    def __init__(self, x=None, y=None) -> None:
        super().__init__()
        self.targets: FloatTensor = y
        self.features: LongTensor = x


class Avazu(TabularDataset):

    def __init__(self, train_path, test_path, split='train') -> None:
        super().__init__()
        self.split = split
        parse_date = lambda val : pd.datetime.strptime(val, '%y%m%d%H')
        if split == 'train' or split == 'valid':
            self.train_data = pd.read_csv(train_path,
                                        parse_dates=['hour'], date_parser=parse_date)
            
            self._preprocess_train()
        elif split == 'test':
            pass

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
        target_encoder = TargetEncoder()
        X = target_encoder.fit_transform(X, self.train_data['y'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state= 42)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        if self.split == 'train':
            self.features = torch.from_numpy(X_train)
            self.targets = torch.from_numpy(y_train)
        elif self.split == 'valid':
            self.features = torch.from_numpy(X_test)
            self.targets = torch.from_numpy(y_test)

        
class Income(TabularDataset):

    def __init__(self, path, split='train') -> None:
        super().__init__()

        self.split = split
        if split == 'train' or 'valid':
            self.data = pd.read_csv(os.path.join(path, 'train.csv'))
            self._preprocess()
        elif split == 'test':
            self.data = pd.read_csv(os.path.join(path, 'test.csv'))
        else:
            raise ValueError(f'Key {split} not known')
        
    def _preprocess(self):
        self.data.rename(columns={'income_>50K': 'income'}, inplace=True)

        # handle outliers
        col = ['capital-loss', 'capital-gain', 'fnlwgt']
        for col in col:
            percentiles = self.data[col].quantile(0.98)
            if self.data[col].quantile(0.98) < 0.5 * self.data[col].max():
                self.data[col][self.data[col] >= percentiles] = percentiles

        X = self.data.drop(['income'], axis=1)
        y = self.data['income']
        target_encoder = TargetEncoder()
        X = target_encoder.fit_transform(X, self.data['income'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state= 42)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        if self.split == 'train':
            self.features = torch.from_numpy(X_train)
            self.targets = torch.from_numpy(y_train.to_numpy())
        elif self.split == 'valid':
            self.features = torch.from_numpy(X_test)
            self.targets = torch.from_numpy(y_test.to_numpy())