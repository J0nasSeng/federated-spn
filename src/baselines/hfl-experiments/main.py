import numpy as np

from fedlab.utils.functional import evaluate as eval_multiclass
from fedlab.core.standalone import StandalonePipeline

from torch import nn
from torch.utils.data import DataLoader
import fedlab.contrib.algorithm as client
import fedlab.contrib.algorithm.basic_server as server
from fedlab.models.mlp import MLP
from fedlab.models.cnn import CNN_MNIST, CNN_CIFAR10
from datasets.utils import get_horizontal_train_data, get_test_dataset
from utils import evaluate_binary
import torch
import argparse
import os
import pandas as pd
from models import AlexNet
from vit_pytorch import SimpleViT
import rtpt

class EvalPipeline(StandalonePipeline):
    def __init__(self, handler, trainer, test_loader, criterion, evaluate, args):
        super().__init__(handler, trainer)
        self.test_loader = test_loader 
        self.loss = []
        self.acc = []
        self.criterion = criterion
        self.evaluate = evaluate
        self.args = args
        self.rtpt = rtpt.RTPT('JS', 'ViT_Baseline', args.comm_rounds)
        self.rtpt.start()
        
    def main(self):
        t=0
        while self.handler.if_stop is False:
            # server side
            sampled_clients = self.handler.sample_clients()
            broadcast = self.handler.downlink_package
            
            # client side
            self.trainer.local_process(broadcast, sampled_clients)
            uploads = self.trainer.uplink_package

            # server side
            for pack in uploads:
                self.handler.load(pack)

            self.rtpt.step()
            loss, acc = self.evaluate(self.handler.model, self.criterion, self.test_loader)
            print("Round {}, Loss {:.4f}, Test Accuracy {:.4f}".format(t, loss, acc))
            t+=1
            self.loss.append(loss)
            self.acc.append(acc)
        
        # log results
        if os.path.isfile('./baseline_experiments.csv'):
            df = pd.read_csv('./baseline_experiments.csv', index_col=0)
            table_dict = df.to_dict()
            table_dict = {k: list(v.values()) for k, v in table_dict.items()}
        else:
            table_dict = {'dataset': [], 'setting': [], 'rounds': [],
                  'clients': [], 'accuracy': [], 'skew': []}
        table_dict['accuracy'].append(self.acc[-1])
        table_dict['clients'].append(self.args.num_clients)
        table_dict['dataset'].append(self.args.dataset)
        table_dict['setting'].append('horizontal')
        table_dict['skew'].append(self.args.partitioning)
        table_dict['rounds'].append(self.args.comm_rounds)
        df = pd.DataFrame.from_dict(table_dict)
        df.to_csv('./baseline_experiments.csv')

def run_pipeline(args):
    if args.partitioning == 'iid':
        args.dir_alpha = 0
    dataset = get_horizontal_train_data(args.dataset, args.num_clients, args.partitioning)
    
    if args.dataset == 'mnist':
        if args.model == 'cnn':
            model = AlexNet(channels=3)
        elif args.model == 'vit':
            model = SimpleViT(
                image_size=28,
                patch_size=8,
                num_classes=10,
                dim=512,
                depth=4,
                heads=16,
                mlp_dim=1024,
                channels=1
            )
    elif args.dataset == 'cifar10':
        if args.model == 'cnn':
            model = AlexNet(channels=3)
        elif args.model == 'vit':
            model = SimpleViT(
                image_size=32,
                patch_size=8,
                num_classes=10,
                dim=512,
                depth=4,
                heads=16,
                mlp_dim=1024,
                channels=3
            )
    else:
        model = nn.Sequential(
            MLP(dataset.in_dim, dataset.out_dim),
            nn.Softmax())

    cuda = torch.cuda.is_available()
    if args.algorithm == 'fedavg':
        trainer = client.FedAvgSerialClientTrainer(model, args.num_clients, cuda, args.gpu)
    elif args.algorithm == 'scaffold':
        trainer = client.ScaffoldSerialClientTrainer(model, args.num_clients, cuda, args.gpu)
    elif args.algorithm == 'fedprox':
        trainer = client.FedProxSerialClientTrainer(model, args.num_clients, cuda, args.gpu)
    trainer.setup_dataset(dataset)
    trainer.setup_optim(args.epochs, args.batch_size, args.lr)

    handler = server.SyncServerHandler(model, args.comm_rounds, args.sample_ratio, cuda, args.gpu)

    test_data = get_test_dataset(args.dataset)
    test_loader = DataLoader(test_data, batch_size=1024)

    if args.dataset == 'mnist' or args.dataset == 'cifar10':
        criterion = nn.CrossEntropyLoss()
        evaluate_fun = eval_multiclass
    else:
        criterion = nn.BCELoss()
        evaluate_fun = evaluate_binary
    standalone_eval = EvalPipeline(handler=handler, trainer=trainer, test_loader=test_loader, criterion=criterion, evaluate=evaluate_fun)
    standalone_eval.main()

parser = argparse.ArgumentParser()

parser.add_argument('--algorithm', default='fedavg')
parser.add_argument('--num-clients', default=2, type=int)
parser.add_argument('--dataset', default='mnist')
parser.add_argument('--gpu', default=None, type=int or None)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--batch-size', default=32, type=int)
parser.add_argument('--partitioning', default='iid')
parser.add_argument('--dir-alpha', default=0.2, type=float)
parser.add_argument('--comm-rounds', default=10, type=int)
parser.add_argument('--sample-ratio', default=1.0, type=float)
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--model', default='vit')

args = parser.parse_args()

run_pipeline(args)