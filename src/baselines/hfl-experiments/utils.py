import torch

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
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(gpu)
            labels = labels.to(gpu)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            _, y_true = torch.max(labels, 1)
            loss_.update(loss.item())
            acc_.update(torch.sum(predicted.eq(y_true)).item(), len(y_true))

    return loss_.sum, acc_.avg