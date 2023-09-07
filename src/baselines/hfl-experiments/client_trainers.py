from fedlab.contrib.algorithm import SGDSerialClientTrainer
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
import torch
from copy import deepcopy

class TabNetFedAvgSerialClientTrainer(SGDSerialClientTrainer):

    def __init__(self, model, num_clients, cuda=False, device=None, logger=None, personal=False) -> None:
        super().__init__(model, num_clients, cuda, device, logger, personal)
        self.optimizer = Adam(self.model.parameters(), 0.02)
        self.lr_scheduler = ExponentialLR(self.optimizer, 0.9)


    """Federated client with local SGD solver."""
    def train(self, model_parameters, train_loader):
        self.set_model(model_parameters)
        self._model.train()

        data_size = 0
        for e in range(self.epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                if self.cuda:
                    data = data.cuda(self.device)
                    target = target.cuda(self.device)

                output, _ = self.model(data)
                loss = self.criterion(output, target)

                data_size += len(target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if e % 10 == 0:
                self.lr_scheduler.step()

        return [self.model_parameters, data_size]
    
class TabNetFedProxSerialClientTrainer(SGDSerialClientTrainer):
    def setup_optim(self, epochs, batch_size, lr, mu):
        super().setup_optim(epochs, batch_size, lr)
        self.mu = mu

    def local_process(self, payload, id_list):
        model_parameters = payload[0]
        for id in id_list:
            data_loader = self.dataset.get_dataloader(id, self.batch_size)
            pack = self.train(model_parameters, data_loader, self.mu)
            self.cache.append(pack)

    def train(self, model_parameters, train_loader, mu) -> None:
        """Client trains its local model on local dataset.

        Args:
            model_parameters (torch.Tensor): serialized model parameters.
            train_loader (torch.utils.data.DataLoader): :class:`torch.utils.data.DataLoader` for this client.
            mu (float): parameter of FedProx.
            
        """
        self.set_model(model_parameters)
        frz_model = deepcopy(self._model)
        frz_model.eval()

        for ep in range(self.epochs):
            self._model.train()
            for data, target in train_loader:
                if self.cuda:
                    data, target = data.cuda(self.device), target.cuda(
                        self.device)

                preds, _ = self._model(data)
                l1 = self.criterion(preds, target)
                l2 = 0.0
                for w0, w in zip(frz_model.parameters(), self._model.parameters()):
                    l2 += torch.sum(torch.pow(w - w0, 2))

                loss = l1 + 0.5 * mu * l2

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return [self.model_parameters]
    
class TabNetScaffoldSerialClientTrainer(SGDSerialClientTrainer):
    def setup_optim(self, epochs, batch_size, lr):
        super().setup_optim(epochs, batch_size, lr)
        self.cs = [None for _ in range(self.num_clients)]

    def local_process(self, payload, id_list):
        model_parameters = payload[0]
        global_c = payload[1]
        for id in id_list:
            data_loader = self.dataset.get_dataloader(id, self.batch_size)
            pack = self.train(id, model_parameters, global_c, data_loader)
            self.cache.append(pack)

    def train(self, id, model_parameters, global_c, train_loader):
        self.set_model(model_parameters)
        frz_model = model_parameters

        if self.cs[id] is None:
            self.cs[id] = torch.zeros_like(model_parameters)

        for _ in range(self.epochs):
            for data, target in train_loader:
                if self.cuda:
                    data = data.cuda(self.device)
                    target = target.cuda(self.device)

                output, _ = self.model(data)
                loss = self.criterion(output, target)

                self.optimizer.zero_grad()
                loss.backward()

                grad = self.model_gradients
                grad = grad - self.cs[id] + global_c
                idx = 0
                for parameter in self._model.parameters():
                    layer_size = parameter.grad.numel()
                    shape = parameter.grad.shape
                    #parameter.grad = parameter.grad - self.cs[id][idx:idx + layer_size].view(parameter.grad.shape) + global_c[idx:idx + layer_size].view(parameter.grad.shape)
                    parameter.grad.data[:] = grad[idx:idx+layer_size].view(shape)[:]
                    idx += layer_size

                self.optimizer.step()

        dy = self.model_parameters - frz_model
        dc = -1.0 / (self.epochs * len(train_loader) * self.lr) * dy - global_c
        self.cs[id] += dc
        return [dy, dc]