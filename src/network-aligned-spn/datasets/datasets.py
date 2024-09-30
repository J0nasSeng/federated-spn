from torch.utils.data import Dataset
from torch import FloatTensor, LongTensor
from sklearn.preprocessing import StandardScaler, TargetEncoder, RobustScaler, LabelEncoder
from sklearn.experimental.enable_iterative_imputer import IterativeImputer
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd 
import torch
import os

class TabularDataset(Dataset):

    def __init__(self, x=None, y=None) -> None:
        super().__init__()
        self.X_train, self.X_valid, self.X_test, self.y_train, self.y_valid, self.y_test = None, None, None, None, None, None
        self.targets: FloatTensor = y
        self.features: FloatTensor = x

    def set_split(self, split):
        if split == 'train':
            self.features = self.X_train
            self.targets = self.y_train
        elif split == 'test':
            self.features = self.X_test
            self.targets = self.y_test
        elif split == 'valid':
            self.features = self.X_valid
            self.targets = self.y_valid


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
        self.name = 'avazu'

        

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

    def __init__(self, path) -> None:
        super().__init__()
        self.name = 'breast-cancer'
        self.data = pd.read_csv(os.path.join(path, 'data.csv'))
        self.X_train, self.X_valid, self.X_test, self.y_train, self.y_valid, self.y_test = self._preprocess()

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
        y_train = torch.from_numpy(y_train)
        y_valid = torch.from_numpy(y_valid)
        y_test = torch.from_numpy(y_test)
        return X_train, X_valid, X_test, y_train, y_valid, y_test

class GimmeCredit(TabularDataset):

    def __init__(self, path) -> None:
        super().__init__()
        self.name = 'credit'
        self.train_data = pd.read_csv(os.path.join(path, 'cs-training.csv'))
        self.X_train, self.X_valid, self.X_test, self.y_train, self.y_valid, self.y_test = self._preprocess()
        
    def _preprocess(self):
        self.train_data = self.train_data[self.train_data['age'] > 21] # filter outliers
        self.train_data = self.train_data.drop(columns=['Unnamed: 0'])
        y = self.train_data['SeriousDlqin2yrs'].to_numpy()
        X = self.train_data.drop(columns=['SeriousDlqin2yrs']).to_numpy()
        X_train, X_valid, y_train, y_valid = train_test_split(X,y,test_size=0.3,random_state=42)
        X_valid, X_test, y_valid, y_test = train_test_split(X_valid, y_valid, test_size=0.5, random_state=42)
        sc = RobustScaler()
        scale_cols = [0, 3 , 4]
        X_train[:, scale_cols] = sc.fit_transform(X_train[:, scale_cols])
        X_valid[:, scale_cols] = sc.transform(X_valid[:, scale_cols])
        X_test[:, scale_cols] = sc.transform(X_test[:, scale_cols])

        imputer = IterativeImputer()
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.fit_transform(X_test)

        # undersample negative examples
        zero_idx = np.argwhere(y_train == 0).flatten()
        one_idx = np.argwhere(y_train == 1).flatten()
        subidx = np.random.choice(zero_idx, int(len(zero_idx) * 0.85), False)
        subidx = np.concatenate((one_idx, subidx))
        X_train, y_train = X_train[subidx], y_train[subidx]

        X_train = torch.from_numpy(X_train)
        X_valid = torch.from_numpy(X_valid)
        X_test = torch.from_numpy(X_test)
        y_train = torch.from_numpy(y_train)
        y_valid = torch.from_numpy(y_valid)
        y_test = torch.from_numpy(y_test)
        return X_train, X_valid, X_test, y_train, y_valid, y_test


class Income(TabularDataset):

    def __init__(self, path) -> None:
        super().__init__()
        self.name = 'income'
        self.train_data = pd.read_csv(os.path.join(path, 'train.csv'))
        self.test_data = pd.read_csv(os.path.join(path, 'train.csv'))
        self.X_train, self.X_valid, self.X_test, self.y_train, self.y_valid, self.y_test = self._preprocess()

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
    

class BAFDataset(TabularDataset):

    def __init__(self, path, scale_all=False):
        super().__init__()
        self.name = 'baf'
        self._scale_all = scale_all
        self.data = pd.read_csv(os.path.join(path, 'Base.csv'))
        self.X_train, self.X_valid, self.X_test, self.y_train, self.y_valid, self.y_test = self._preprocess()

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        x = self.features[index]
        y = self.targets[index]
        return torch.hstack([x, y.unsqueeze(0)])

    def _preprocess(self):
        y = self.data['fraud_bool']
        x = self.data.drop(columns=['fraud_bool', 'device_fraud_count'])
        col_types = infer_column_types(x)
        columns = x.columns.tolist() + ['fraud_bool']
        # TODO: replace with TargetEncoder to make compatible to einets (they only allow Gaussians in leafs currently)
        label_encoder = LabelEncoder()
        for col in x.select_dtypes(include=['object']).columns:
            x[col] = label_encoder.fit_transform(x[col])

        y = y.to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.3, random_state= 42)

        train_data_np = np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1)
        test_data_np = np.concatenate((X_test, y_test.reshape(-1, 1)), axis=1)
        self.train_data = pd.DataFrame(train_data_np, columns=columns)
        self.test_data = pd.DataFrame(test_data_np, columns=columns)

        # handle outliers
        col = ['prev_address_months_count', 'days_since_request', 'intended_balcon_amount']
        for col in col:
            percentiles = self.train_data[col].quantile(0.98)
            if self.train_data[col].quantile(0.98) < 0.5 * self.train_data[col].max():
                self.train_data[col][self.train_data[col] >= percentiles] = percentiles
                self.test_data[col][self.test_data[col] >= percentiles] = percentiles

        X = self.train_data.drop(['fraud_bool'], axis=1)
        y = self.train_data['fraud_bool']
        X_test = self.test_data.drop(['fraud_bool'], axis=1)
        y_test = self.test_data['fraud_bool']
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify=y, test_size=0.3, random_state= 42)

        X_train, X_valid, X_test = X_train.to_numpy(), X_valid.to_numpy(), X_test.to_numpy()

        scale_cols = [i for i, c in col_types.items() if c=='continuous']
        sc = StandardScaler()
        if self._scale_all:
            X_train = sc.fit_transform(X_train)
            X_valid = sc.transform(X_valid)
            X_test = sc.transform(X_test)
        else:
            X_train[:,scale_cols] = sc.fit_transform(X_train[:, scale_cols])
            X_valid[:,scale_cols] = sc.transform(X_valid[:, scale_cols])
            X_test[:,scale_cols] = sc.transform(X_test[:, scale_cols])

        X_train = torch.from_numpy(X_train)
        X_valid = torch.from_numpy(X_valid)
        X_test = torch.from_numpy(X_test)
        y_train = torch.from_numpy(y_train.to_numpy())
        y_valid = torch.from_numpy(y_valid.to_numpy())
        y_test = torch.from_numpy(y_test.to_numpy())
        return X_train, X_valid, X_test, y_train, y_valid, y_test

class AirlinesDataset(TabularDataset):

    def __init__(self, path, scale_all=False):
        super().__init__()
        self.name = 'airlines'
        self._scale_all = scale_all
        self.data = pd.read_csv(os.path.join(path, 'Base.csv'))
        self.X_train, self.X_valid, self.X_test, self.y_train, self.y_valid, self.y_test = self._preprocess()

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        x = self.features[index]
        y = self.targets[index]
        return torch.hstack([x, y.unsqueeze(0)])

    def _preprocess(self):
        y = self.data['fraud_bool']
        x = self.data.drop(columns=['fraud_bool', 'device_fraud_count'])
        col_types = infer_column_types(x)
        columns = x.columns.tolist() + ['fraud_bool']
        # TODO: replace with TargetEncoder to make compatible to einets (they only allow Gaussians in leafs currently)
        label_encoder = LabelEncoder()
        for col in x.select_dtypes(include=['object']).columns:
            x[col] = label_encoder.fit_transform(x[col])

        y = y.to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.3, random_state= 42)

        train_data_np = np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1)
        test_data_np = np.concatenate((X_test, y_test.reshape(-1, 1)), axis=1)
        self.train_data = pd.DataFrame(train_data_np, columns=columns)
        self.test_data = pd.DataFrame(test_data_np, columns=columns)

        # handle outliers
        col = ['prev_address_months_count', 'days_since_request', 'intended_balcon_amount']
        for col in col:
            percentiles = self.train_data[col].quantile(0.98)
            if self.train_data[col].quantile(0.98) < 0.5 * self.train_data[col].max():
                self.train_data[col][self.train_data[col] >= percentiles] = percentiles
                self.test_data[col][self.test_data[col] >= percentiles] = percentiles

        X = self.train_data.drop(['fraud_bool'], axis=1)
        y = self.train_data['fraud_bool']
        X_test = self.test_data.drop(['fraud_bool'], axis=1)
        y_test = self.test_data['fraud_bool']
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify=y, test_size=0.3, random_state= 42)

        X_train, X_valid, X_test = X_train.to_numpy(), X_valid.to_numpy(), X_test.to_numpy()

        scale_cols = [i for i, c in col_types.items() if c=='continuous']
        sc = StandardScaler()
        if self._scale_all:
            X_train = sc.fit_transform(X_train)
            X_valid = sc.transform(X_valid)
            X_test = sc.transform(X_test)
        else:
            X_train[:,scale_cols] = sc.fit_transform(X_train[:, scale_cols])
            X_valid[:,scale_cols] = sc.transform(X_valid[:, scale_cols])
            X_test[:,scale_cols] = sc.transform(X_test[:, scale_cols])

        X_train = torch.from_numpy(X_train)
        X_valid = torch.from_numpy(X_valid)
        X_test = torch.from_numpy(X_test)
        y_train = torch.from_numpy(y_train.to_numpy())
        y_valid = torch.from_numpy(y_valid.to_numpy())
        y_test = torch.from_numpy(y_test.to_numpy())
        return X_train, X_valid, X_test, y_train, y_valid, y_test
    
class SantanderDataset(TabularDataset):

    def __init__(self, path, scale_all=False):
        super().__init__()
        self.name = 'santander'
        self._scale_all = scale_all
        self.train_data = pd.read_csv(os.path.join(path, 'train.csv'))
        self.test_data = pd.read_csv(os.path.join(path, 'test.csv'))
        self.X_train, self.X_valid, self.X_test, self.y_train, self.y_valid, self.y_test = self._preprocess()

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        x = self.features[index]
        y = self.targets[index]
        return torch.hstack([x, y.unsqueeze(0)])

    def _preprocess(self):
        y = self.train_data['target']
        x = self.train_data.drop(columns=['target', 'ID_code'])
        columns = x.columns.tolist() + ['target']

        y = y.to_numpy()
        X_train, X_valid, y_train, y_valid = train_test_split(x, y, stratify=y, test_size=0.3, random_state= 42)

        self.train_data = pd.DataFrame(X_train, columns=columns)
        self.val_data = pd.DataFrame(X_valid, columns=columns)

        self.test_data = self.test_data.drop(columns=['ID_code'])

        X_train, X_valid, X_test = X_train.to_numpy(), X_valid.to_numpy(), self.test_data.to_numpy()

        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_valid = sc.transform(X_valid)
        X_test = sc.transform(X_test)

        X_train = torch.from_numpy(X_train)
        X_valid = torch.from_numpy(X_valid)
        X_test = torch.from_numpy(X_test)
        y_train = torch.from_numpy(y_train)
        y_valid = torch.from_numpy(y_valid)
        return X_train, X_valid, X_test, y_train, y_valid, y_valid # NOTE: targets only returned for compatibility reasons, not used

class DatasetFactory:

    def __init__(self):
        self.loaded_datasets = {}

    def load_dataset(self, ds, **ds_kwargs) -> Dataset:
        if ds in self.loaded_datasets.keys():
            return self.loaded_datasets[ds]
        if ds == 'income':
            dataset = Income('../../datasets/income/')
        elif ds == 'breast-cancer':
            dataset = BreastCancer('../../datasets/breast-cancer/')
        elif ds == 'credit':
            dataset = GimmeCredit('../../datasets/GiveMeSomeCredit/')
        elif ds == 'baf':
            dataset = BAFDataset('../../datasets/BAF/', **ds_kwargs)
        elif ds == 'santander':
            dataset = SantanderDataset('../../datasets/santander', **ds_kwargs)
        self.loaded_datasets[ds] = dataset
        
        return dataset
    
    
def infer_column_types(df):
    column_types = {}
    
    for i, col in enumerate(df.columns):
        unique_vals = df[col].nunique()
        total_vals = len(df[col])
        dtype = df[col].dtype

        # Categorical: Object types or integers with a small number of unique values
        if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype):
            if unique_vals == 2:
                column_types[i] = 'binary'
            else:
                column_types[i] = 'categorical'
        elif pd.api.types.is_integer_dtype(dtype):
            # Heuristic: Consider discrete if unique values are small relative to total rows
            if unique_vals / total_vals < 0.05:  # Arbitrary threshold
                if unique_vals == 2:
                    column_types[i] = 'binary'
                else:
                    column_types[i] = 'discrete'
            else:
                column_types[i] = 'continuous'
        elif pd.api.types.is_float_dtype(dtype):
            # Heuristic: Assume float columns are continuous
            if unique_vals == 2:
                column_types[i] = 'binary'
            else:
                column_types[i] = 'continuous'
        else:
            column_types[i] = 'unknown'
    
    return column_types