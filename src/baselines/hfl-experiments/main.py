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

class EvalPipeline(StandalonePipeline):
    def __init__(self, handler, trainer, test_loader, criterion, evaluate):
        super().__init__(handler, trainer)
        self.test_loader = test_loader 
        self.loss = []
        self.acc = []
        self.criterion = criterion
        self.evaluate = evaluate
        
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

            loss, acc = self.evaluate(self.handler.model, self.criterion, self.test_loader)
            print("Round {}, Loss {:.4f}, Test Accuracy {:.4f}".format(t, loss, acc))
            t+=1
            self.loss.append(loss)
            self.acc.append(acc)
        
def run_pipeline(args):
    dataset = get_horizontal_train_data(args.dataset, args.num_clients, args.partitioning)
    
    if args.dataset == 'mnist':
        model = CNN_MNIST()
    elif args.dataset == 'cifar10':
        model = CNN_CIFAR10()
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
parser.add_argument('--comm-rounds', default=10, type=int)
parser.add_argument('--sample-ratio', default=1.0, type=float)
parser.add_argument('--lr', default=0.01, type=float)

args = parser.parse_args()

run_pipeline(args)