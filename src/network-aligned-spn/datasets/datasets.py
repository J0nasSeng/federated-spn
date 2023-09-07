from torch.utils.data import Dataset
from torch import FloatTensor, LongTensor
from sklearn.preprocessing import StandardScaler, TargetEncoder, RobustScaler
from sklearn.experimental.enable_iterative_imputer import IterativeImputer
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd 
import torch
import os
from datetime import datetime

class TabularDataset(Dataset):

    def __init__(self, x=None, y=None) -> None:
        super().__init__()
        self.targets: FloatTensor = y
        self.features: LongTensor = x


class Avazu(TabularDataset):

    def __init__(self, train_path, split='train') -> None:
        super().__init__()
        self.split = split
        if split == 'train' or split == 'valid':
            self.train_data = pd.read_csv(os.path.join(train_path, 'train.csv'),
                                        parse_dates=['hour'], date_format='%y%m%d%H')
            
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
            self.targets = torch.from_numpy(y_train.to_numpy())
        elif self.split == 'valid':
            self.features = torch.from_numpy(X_test)
            self.targets = torch.from_numpy(y_test.to_numpy())

class BreastCancer(TabularDataset):

    def __init__(self, path, split='train') -> None:
        super().__init__()
        self.data = pd.read_csv(os.path.join(path, 'data.csv'))
        X_train, X_valid, X_test, y_train, y_valid, y_test = self._preprocess()
        if split == 'train':
            self.features = X_train
            self.targets = y_train
        elif split == 'test':
            self.features = X_test
            self.targets = y_test
        elif split == 'valid':
            self.features = X_valid
            self.targets = y_valid

    def _preprocess(self):
        self.data.drop(columns=['id'], inplace=True)
        y = self.data['diagnosis']
        x = self.data.drop(columns=['diagnosis'])
        y.iloc[y == 'M'] = 1
        y.iloc[y == 'B'] = 0
        x, y = x.to_numpy(), y.to_numpy()

        X_train, X_valid, y_train, y_valid = train_test_split(x, y, stratify=y, test_size=0.3, random_state= 42)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_valid = sc.transform(X_valid)
        X_test = sc.transform(X_test)

        X_train = torch.from_numpy(X_train)
        X_valid = torch.from_numpy(X_valid)
        X_test = torch.from_numpy(X_test)
        y_train = torch.from_numpy(y_train.to_numpy())
        y_valid = torch.from_numpy(y_valid.to_numpy())
        y_test = torch.from_numpy(y_test.to_numpy())
        return X_train, X_valid, X_test, y_train, y_valid, y_test

class GimmeCredit(TabularDataset):

    def __init__(self, path, split='train') -> None:
        super().__init__()
        self.train_data = pd.read_csv(os.path.join(path, 'cs-training.csv'))
        self.test_data = pd.read_csv(os.path.join(path, 'cs-test.csv'))
        X_train, X_valid, X_test, y_train, y_valid, y_test = self._preprocess()
        if split == 'train':
            self.features = X_train
            self.targets = y_train
        elif split == 'test':
            self.features = X_test
            self.targets = y_test
        elif split == 'valid':
            self.features = X_valid
            self.targets = y_valid
        
    def _preprocess(self):
        self.train_data = self.train_data[self.train_data['age'] > 21] # filter outliers 
        y = self.train_data['SeriousDlqin2yrs'].to_numpy()
        X = self.train_data.drop(columns=['SeriousDlqin2yrs']).to_numpy()
        y_test = self.test_data['SeriousDlqin2yrs'].to_numpy()
        X_test = self.test_data.drop(columns=['SeriousDlqin2yrs'])
        X_train, X_valid, y_train, y_valid = train_test_split(X,y,test_size=0.3,random_state=42)
        sc = RobustScaler()
        X_train = sc.fit_transform(X_train)
        X_valid = sc.transform(X_valid)
        X_test = sc.transform(X_test)

        imputer = IterativeImputer()
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.fit_transform(X_test)

        X_train = torch.from_numpy(X_train)
        X_valid = torch.from_numpy(X_valid)
        X_test = torch.from_numpy(X_test)
        y_train = torch.from_numpy(y_train)
        y_valid = torch.from_numpy(y_valid)
        y_test = torch.from_numpy(y_test)
        return X_train, X_valid, X_test, y_train, y_valid, y_test


class Income(TabularDataset):

    def __init__(self, path, split='train') -> None:
        super().__init__()

        self.split = split
        self.train_data = pd.read_csv(os.path.join(path, 'train.csv'))
        self.test_data = pd.read_csv(os.path.join(path, 'train.csv'))
        X_train, X_valid, X_test, y_train, y_valid, y_test = self._preprocess()
        if split == 'train':
            self.features = X_train
            self.targets = y_train
        elif split == 'valid':
            self.features = X_valid
            self.targets = y_valid
        elif split == 'test':
            self.features = X_test
            self.targets = y_test

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        x = self.features[index]
        y = self.targets[index]
        return torch.hstack([x, y.unsqueeze(0)])
        
    def _preprocess(self):
        self.train_data.rename(columns={'income_>50K': 'income'}, inplace=True)
        self.test_data.rename(columns={'income_>50K': 'income'}, inplace=True)

        # handle outliers
        col = ['capital-loss', 'capital-gain', 'fnlwgt']
        for col in col:
            percentiles = self.train_data[col].quantile(0.98)
            if self.train_data[col].quantile(0.98) < 0.5 * self.train_data[col].max():
                self.train_data[col][self.train_data[col] >= percentiles] = percentiles
                self.test_data[col][self.test_data[col] >= percentiles] = percentiles

        X = self.train_data.drop(['income'], axis=1)
        y = self.train_data['income']
        X_test = self.test_data.drop(['income'], axis=1)
        y_test = self.test_data['income']
        target_encoder = TargetEncoder()
        X = target_encoder.fit_transform(X, self.train_data['income'])
        X_test = target_encoder.transform(X_test)
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify=y, test_size=0.3, random_state= 42)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_valid = sc.transform(X_valid)
        X_test = sc.transform(X_test)

        X_train = torch.from_numpy(X_train)
        X_valid = torch.from_numpy(X_valid)
        X_test = torch.from_numpy(X_test)
        y_train = torch.from_numpy(y_train.to_numpy())
        y_valid = torch.from_numpy(y_valid.to_numpy())
        y_test = torch.from_numpy(y_test.to_numpy())
        return X_train, X_valid, X_test, y_train, y_valid, y_test