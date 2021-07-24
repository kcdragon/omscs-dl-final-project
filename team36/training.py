import copy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

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


def predict(model, input):
    inputs = input.unsqueeze(0)
    out = model(inputs)
    _, prediction = torch.max(out, dim=-1)
    return prediction.item()


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



def do_training(model, training_split, validation_split, epochs=2, learning_rate=1e-3, momentum=5e-1, weight_decay=5e-2, batch_size=128):
    """Do the full training/validation loop and generate graphs"""
    sampler = torch.utils.data.RandomSampler(training_split, replacement=True, num_samples=1000)
    training_loader = torch.utils.data.DataLoader(training_split, batch_size=batch_size, sampler=sampler)
    test_loader = torch.utils.data.DataLoader(validation_split, batch_size=batch_size, shuffle=False, num_workers=2)

    if torch.cuda.is_available():
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), learning_rate,
                                momentum=momentum, weight_decay=weight_decay)

    best = 0.0
    best_cm = None
    best_model = None
    train_accuracy_history = []
    train_loss_history = []
    validation_accuracy_history = []
    validation_loss_history = []
    for epoch in range(epochs):
        train_acc, train_loss = train(epoch, training_loader, model, optimizer, criterion)
        train_accuracy_history.append(train_acc)
        train_loss_history.append(train_loss)

        acc, cm, loss = validate(epoch, test_loader, model, criterion)
        validation_accuracy_history.append(acc)
        validation_loss_history.append(loss)

        print("Epoch {0} | Training accuracy: {1}% | Validation accuracy: {2}%".format(epoch, train_acc, acc))

        if acc > best:
            best = acc
            best_cm = cm
            best_model = copy.deepcopy(model)

    training_curve, = plt.plot(train_accuracy_history, label='training')
    validation_curve, = plt.plot(validation_accuracy_history, label='validation')
    plt.title('Accuracy Curve')
    plt.legend(handles=[training_curve, validation_curve])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()

    training_curve, = plt.plot(train_loss_history, label='training')
    validation_curve, = plt.plot(validation_loss_history, label='validation')
    plt.title('Loss Curve')
    plt.legend(handles=[training_curve, validation_curve])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    print('Best Validation Acccuracy: {:.4f}'.format(best))