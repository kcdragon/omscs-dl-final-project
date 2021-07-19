import torch


# source: Assignment 2
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# source: Assignment 2
def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.shape[0]

    _, pred = torch.max(output, dim=-1)

    correct = pred.eq(target).sum() * 1.0

    acc = correct / batch_size

    return acc


# source: Assignment 2
def train(epoch, data_loader, model, optimizer, criterion):
    losses = AverageMeter()
    acc = AverageMeter()

    for idx, (data, target) in enumerate(data_loader):
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        out = model(data)
        loss = criterion(out, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_acc = accuracy(out, target)

        losses.update(loss, out.shape[0])
        acc.update(batch_acc, out.shape[0])

        if idx % 10 == 0:
            print(('Epoch: [{0}][{1}/{2}]\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Prec @1 {top1.val:.4f} ({top1.avg:.4f})\t')
                  .format(epoch, idx, len(data_loader), loss=losses, top1=acc))

    return acc.avg, losses.avg.item()


# source: Assignment 2
def validate(epoch, val_loader, model, criterion):
    losses = AverageMeter()
    acc = AverageMeter()

    num_class = 10
    cm = torch.zeros(num_class, num_class)
    for idx, (data, target) in enumerate(val_loader):
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        with torch.no_grad():
            out = model(data)
            loss = criterion(out, target)

        batch_acc = accuracy(out, target)

        # update confusion matrix
        _, preds = torch.max(out, 1)
        for t, p in zip(target.view(-1), preds.view(-1)):
            cm[t.long(), p.long()] += 1

        losses.update(loss, out.shape[0])
        acc.update(batch_acc, out.shape[0])

    cm = cm / cm.sum(1)
    return acc.avg, cm, losses.avg
