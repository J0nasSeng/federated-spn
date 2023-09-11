from torch.utils.data import Dataset, TensorDataset
from torch import FloatTensor, LongTensor
from sklearn.preprocessing import StandardScaler, TargetEncoder, RobustScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.experimental.enable_iterative_imputer import IterativeImputer
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

        X = self.train_data.drop(['income'], axis=1).to_numpy()
        y = self.train_data['income'].to_numpy()
        types = self.train_data.dtypes
        cols = self.train_data.columns
        self.cat_cols = []
        for i in range(X.shape[1]):
            if types[cols[i]] == 'object' or len(np.unique(X[:, i])) < 200:
                le = LabelEncoder()
                X[:, i] = le.fit_transform(X[:, i])
                self.cat_cols.append(i)
        X = X.astype(np.float32)
        X_train, X_tv, y_train, y_tv = train_test_split(X, y, stratify=y, test_size=0.3, random_state= 42)
        scale_cols = [i for i in range(X_train.shape[1]) if i not in self.cat_cols]
        X_train[:, scale_cols] = self.sc.fit_transform(X_train[:, scale_cols])
        X_tv[:, scale_cols] = self.sc.transform(X_tv[:, scale_cols])

        X_valid, X_test, y_valid, y_test = train_test_split(X_tv, y_tv, stratify=y_tv, test_size=0.5, random_state=42)

        # undersample negative examples
        zero_idx = np.argwhere(y_train == 0).flatten()
        one_idx = np.argwhere(y_train == 1).flatten()
        subidx = np.random.choice(zero_idx, int(len(zero_idx) * 0.4), False)
        subidx = np.concatenate((one_idx, subidx))
        X_train, y_train = X_train[subidx], y_train[subidx]

        if self.split == 'train':
            self.features = torch.from_numpy(X_train).to(torch.float32)
            self.targets = torch.from_numpy(y_train)
            self.dataset = TensorDataset(self.features, self.targets)
        elif self.split == 'valid':
            self.features = torch.from_numpy(X_valid).to(torch.float32)
            self.targets = torch.from_numpy(y_valid)
            self.dataset = TensorDataset(self.features, self.targets)
        elif self.split == 'test':
            self.features = torch.from_numpy(X_test).to(torch.float32)
            self.targets = torch.from_numpy(y_test)
            self.dataset = TensorDataset(self.features, self.targets)

class BreastCancer(TabularDataset):

    def __init__(self, path, split='train') -> None:
        super().__init__()
        self.data = pd.read_csv(os.path.join(path, 'data.csv'))
        X_train, X_valid, X_test, y_train, y_valid, y_test = self._preprocess()
        if split == 'train':
            self.features = X_train
            self.targets = y_train
            self.dataset = TensorDataset(self.features, self.targets)
        elif split == 'test':
            self.features = X_test
            self.targets = y_test
            self.dataset = TensorDataset(self.features, self.targets)
        elif split == 'valid':
            self.features = X_valid
            self.targets = y_valid
            self.dataset = TensorDataset(self.features, self.targets)

    def _preprocess(self):
        self.data.drop(columns=['id'], inplace=True)
        y = self.data['diagnosis']
        x = self.data.drop(columns=['diagnosis'])
        y.iloc[y == 'M'] = 1
        y.iloc[y == 'B'] = 0
        x, y = x.to_numpy().astype(np.float32), y.to_numpy().astype(np.float32)

        X_train, X_valid, y_train, y_valid = train_test_split(x, y, stratify=y, test_size=0.3, random_state= 42)
        X_valid, X_test, y_valid, y_test = train_test_split(X_valid, y_valid, test_size=0.5, random_state=42)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_valid = sc.transform(X_valid)
        X_test = sc.transform(X_test)

        imputer = IterativeImputer()
        X_train = imputer.fit_transform(X_train)
        X_valid = imputer.fit_transform(X_valid)
        X_test = imputer.fit_transform(X_test)

        X_train = torch.from_numpy(X_train)
        X_valid = torch.from_numpy(X_valid)
        X_test = torch.from_numpy(X_test)
        y_train = torch.from_numpy(y_train).to(dtype=torch.long)
        y_valid = torch.from_numpy(y_valid).to(dtype=torch.long)
        y_test = torch.from_numpy(y_test).to(dtype=torch.long)
        return X_train, X_valid, X_test, y_train, y_valid, y_test

class GimmeCredit(TabularDataset):

    def __init__(self, path, split='train') -> None:
        super().__init__()
        self.train_data = pd.read_csv(os.path.join(path, 'cs-training.csv'))
        X_train, X_valid, X_test, y_train, y_valid, y_test = self._preprocess()
        if split == 'train':
            self.features = X_train
            self.targets = y_train
            self.dataset = TensorDataset(self.features, self.targets)
        elif split == 'test':
            self.features = X_test
            self.targets = y_test
            self.dataset = TensorDataset(self.features, self.targets)
        elif split == 'valid':
            self.features = X_valid
            self.targets = y_valid
            self.dataset = TensorDataset(self.features, self.targets)
        
    def _preprocess(self):
        self.train_data = self.train_data[self.train_data['age'] > 21] # filter outliers
        self.train_data = self.train_data.drop(columns=['Unnamed: 0'])
        y = self.train_data['SeriousDlqin2yrs'].to_numpy()
        X = self.train_data.drop(columns=['SeriousDlqin2yrs']).to_numpy()
        X_train, X_valid, y_train, y_valid = train_test_split(X,y,test_size=0.3,random_state=42)
        X_valid, X_test, y_valid, y_test = train_test_split(X_valid, y_valid, test_size=0.5, random_state=42)
        scale_cols = [0, 3, 4]
        sc = RobustScaler()
        X_train[:, scale_cols] = sc.fit_transform(X_train[:, scale_cols])
        X_valid[:, scale_cols] = sc.transform(X_valid[:, scale_cols])
        X_test[:, scale_cols] = sc.transform(X_test[:, scale_cols])

        imputer = SimpleImputer(strategy='median')
        X_train = imputer.fit_transform(X_train)
        X_valid = imputer.transform(X_valid)
        X_test = imputer.transform(X_test)

        # label encoding for categorical features
        for i in [1, 2, 5, 6, 7, 8, 9]:
            le = LabelEncoder()
            X_ = np.concatenate([X_train[:, i], X_valid[:, i], X_test[:, i]])
            le.fit(X_)
            X_train[:, i] = le.transform(X_train[:, i])
            X_valid[:, i] = le.transform(X_valid[:, i])
            X_test[:, i] = le.transform(X_test[:, i])
            
        # undersample negative examples
        zero_idx = np.argwhere(y_train == 0).flatten()
        one_idx = np.argwhere(y_train == 1).flatten()
        subidx = np.random.choice(zero_idx, int(len(zero_idx) * 0.1), False)
        subidx = np.concatenate((one_idx, subidx))
        X_train, y_train = X_train[subidx], y_train[subidx]

        X_train = torch.from_numpy(X_train)
        X_valid = torch.from_numpy(X_valid)
        X_test = torch.from_numpy(X_test)
        y_train = torch.from_numpy(y_train)
        y_valid = torch.from_numpy(y_valid)
        y_test = torch.from_numpy(y_test)
        return X_train, X_valid, X_test, y_train, y_valid, y_test