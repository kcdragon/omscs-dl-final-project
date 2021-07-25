import copy, os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from team36.mnist.data_loading import MNIST_Loader

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


def predict(model, input, unsqueeze_dim=0):
    inputs = input.unsqueeze(unsqueeze_dim)
    out = model(inputs)
    _, prediction = torch.max(out, dim=-1)
    return prediction.item()

def predict_from_loader(model, data_loader):
    all_out = []
    for idx, (data, target) in enumerate(data_loader):
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        with torch.no_grad():
            out = model(data)
            all_out.append(out)
    out = torch.cat(all_out)
    return out

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
    
def train_val_split(data, test_size=0.1, shuffle=True):
    all_indices = range(len(data))
    train_indices, val_indices, _, _ = train_test_split(
        all_indices,
        data.targets,
        stratify=data.targets,
        test_size=test_size,
        shuffle=shuffle 
    )
    train_split = torch.utils.data.Subset(data, train_indices)
    val_split = torch.utils.data.Subset(data, val_indices)

#         print(f"{len(training_split)} in training set")
#         print(f"{len(validation_split)} in validation set")

    return train_split, val_split
    
def load_or_train(model, checkpoint, DIR='.', DATA_DIR=None, dataset=None, **training_kwargs):
    """
    Load model weights from existing checkpoint, or train the model if no checkpoint exists
    @param model
    @param checkpoint: name of checkpoint (not full path)
    @param train_data: (optional) dataset (if None, will use MNIST)
    @param training_kwargs: any kwargs to pass to do_training() function
    """
    if DATA_DIR is None:
        DATA_DIR = f'{DIR}/data'
    checkpoint_path = f"{DIR}/checkpoints/{checkpoint}"
    if os.path.exists(checkpoint_path): # if trained checkpoint exists, load it
        state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
    else: # else, train the model
        if dataset:
            do_training(model, train_data)
            train_split, val_split = train_val_split(dataset, shuffle=False)
        else: # if no dataset given, use MNIST
            mnist_loader = MNIST_Loader(DIR, DATA_DIR)
            train_split, val_split = mnist_loader.train_val_split(shuffle=False)
        print(f"{len(train_split)} in training set")
        print(f"{len(val_split)} in validation set")
        do_training(model, training_split=train_split, validation_split=val_split, epochs=2)
    torch.save(model.state_dict(), checkpoint_path)