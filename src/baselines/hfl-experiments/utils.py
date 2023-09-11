import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

class AverageMeter(object):
    """Record metrics information"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

def evaluate_binary(model, criterion, test_loader):
    """Evaluate classify task model accuracy.
    
    Returns:
        (loss.sum, acc.avg)
    """
    model.eval()
    gpu = next(model.parameters()).device

    loss_ = AverageMeter()
    acc_ = AverageMeter()
    f1_micro_ = AverageMeter()
    f1_macro_ = AverageMeter()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(gpu)
            labels = labels.to(gpu)

            outputs = model(inputs)
            if isinstance(outputs, tuple):
                # then output is from TabNet
                outputs, _ = outputs
            
            outputs = torch.softmax(outputs, dim=1)
            labels_ = F.one_hot(labels).to(torch.float32)
            loss = criterion(outputs, labels_)

            _, predicted = torch.max(outputs, 1)
            _, y_true = torch.max(labels_, 1)
            loss_.update(loss.item())
            acc_.update(torch.sum(predicted.eq(y_true)).item(), len(y_true))
            
            f1_micro = f1_score(labels.numpy(), predicted.numpy(), average='micro')
            f1_macro = f1_score(labels.numpy(), predicted.numpy(), average='macro')
            f1_micro_.update(f1_micro)
            f1_macro_.update(f1_macro)

    return loss_.sum, acc_.avg, f1_micro_.avg, f1_macro_.avg