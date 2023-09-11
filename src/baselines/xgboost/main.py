from datasets.utils import get_train_data, get_test_data
from xgboost import XGBClassifier
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import os

def save_results(args, acc, f1_micro, f1_macro):
    if os.path.isfile('./experiments.csv'):
        df = pd.read_csv('./experiments.csv', index_col=0)
        table_dict = df.to_dict()
        table_dict = {k: list(v.values()) for k, v in table_dict.items()}
    else:
        table_dict = {'dataset': [], 'accuracy': [], 'f1_micro': [], 'f1_macro': []}
    table_dict['accuracy'].append(acc)
    table_dict['dataset'].append(args.dataset)
    table_dict['f1_micro'].append(f1_micro)
    table_dict['f1_macro'].append(f1_macro)
    df = pd.DataFrame.from_dict(table_dict)
    df.to_csv('./experiments.csv')

def main(args):
    print(f"Learn XGBoost on {args.dataset}")
    train_data = get_train_data(args.dataset)
    test_data = get_test_data(args.dataset)
    X, y = train_data[:, :-1], train_data[:, -1]
    objective = 'binary:logistic' if args.objective == 'bce' else 'multi:softprob'
    xgb = XGBClassifier(n_estimators=args.estimators, max_depth=args.max_depth, objective=objective)
    xgb.fit(X, y)

    y_true = test_data[:, -1]
    X_test = test_data[:, :-1]
    y_pred = xgb.predict(X_test)
    acc = accuracy_score(y_true, y_pred)
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_macro = f1_score(y_true, y_pred, average='macro')

    save_results(args, acc, f1_micro, f1_macro)

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default='mnist')
parser.add_argument('--estimators', default=100, type=int)
parser.add_argument('--max-depth', default=3, type=int)
parser.add_argument('--objective', default='bce')
parser.add_argument('--num-experiments', default=1, type=int)

args = parser.parse_args()

for _ in range(args.num_experiments):
    main(args)