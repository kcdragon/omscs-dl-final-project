import torch


# "fast gradient sign method" from EXPLAINING AND HARNESSING ADVERSARIAL EXAMPLES
# gradient calculation from
#   https://stackoverflow.com/questions/54754153/autograd-grad-for-tensor-in-pytorch
class FastSignGradientAttackDataSet(torch.utils.data.Dataset):
    def __init__(self, dataset, model, criterion, epsilon):
        self.dataset = dataset
        self.model = model
        self.criterion = criterion
        self.epsilon = epsilon

    def __getitem__(self, index):
        input, target = self.dataset[index]

        inputs = input.unsqueeze(0)
        inputs.requires_grad_()
        targets = torch.tensor([target])

        out = self.model(inputs)
        loss = self.criterion(out, targets)
        loss_gradient = torch.autograd.grad(outputs=loss, inputs=inputs)

        eta = self.epsilon * torch.sign(loss_gradient[0][0])
        adversarial_input = input + eta

        # normalize to [0, 1] range
        min = torch.min(adversarial_input)
        max = torch.max(adversarial_input)
        adversarial_input = (adversarial_input - min) / (max - min)

        return adversarial_input, target

    def __len__(self):
        return len(self.dataset)
