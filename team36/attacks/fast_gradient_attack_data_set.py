import torch


# "fast gradient sign method" from EXPLAINING AND HARNESSING ADVERSARIAL EXAMPLES
# gradient calculation from
#   https://stackoverflow.com/questions/54754153/autograd-grad-for-tensor-in-pytorch
class FastSignGradientAttackDataSet(torch.utils.data.Dataset):
    def __init__(self, baseline_dataset, model, criterion, epsilon):
        self.baseline_dataset = baseline_dataset
        self.model = model
        self.criterion = criterion
        self.epsilon = epsilon

    def __getitem__(self, index):
        input, target = self.baseline_dataset[index]
        batch = [(input, target)]
        inputs = torch.zeros((len(batch),) + batch[0][0].shape)
        targets = torch.zeros((len(batch)), dtype=torch.long)
        for i, x in enumerate(batch):
            inputs[i] = x[0]
            targets[i] = x[1]
        inputs.requires_grad_()

        out = self.model(inputs)
        loss = self.criterion(out, targets)
        loss_gradient = torch.autograd.grad(outputs=loss, inputs=inputs)

        eta = self.epsilon * torch.sign(loss_gradient[0])
        adversarial_inputs = inputs + eta

        for index in range(adversarial_inputs.shape[0]):
            adversarial_input = adversarial_inputs[index]
            min = torch.min(adversarial_input)
            max = torch.max(adversarial_input)
            adversarial_input = (adversarial_input - min) / (max - min)
        return adversarial_input, target

    def __len__(self):
        return len(self.baseline_dataset)
