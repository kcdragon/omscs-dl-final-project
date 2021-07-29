import torch


# "fast gradient sign method" from EXPLAINING AND HARNESSING ADVERSARIAL EXAMPLES
# gradient calculation from
#   https://stackoverflow.com/questions/54754153/autograd-grad-for-tensor-in-pytorch
class FastSignGradientAttackDataSet(torch.utils.data.Dataset):
    def __init__(self, baseline_dataset, model, criterion, epsilon, device=None):
        self.baseline_dataset = baseline_dataset
        self.model = model
        self.criterion = criterion
        self.epsilon = epsilon
        self.device = device

    def __getitem__(self, index):
        input, target = self.baseline_dataset[index]

        if self.device is not None:
            input = input.to(self.device)

        inputs = input.unsqueeze(0)
        inputs.requires_grad_()
        targets = torch.tensor([target]).to(inputs.device)

        out = self.model(inputs)
        loss = self.criterion(out, targets)
        loss_gradient = torch.autograd.grad(outputs=loss, inputs=inputs)

        eta = self.epsilon * torch.sign(loss_gradient[0])
        adversarial_inputs = inputs + eta

        adversarial_input = adversarial_inputs[0]
        min = torch.min(adversarial_input)
        max = torch.max(adversarial_input)
        adversarial_input = (adversarial_input - min) / (max - min)

        return adversarial_input, target

    def __len__(self):
        return len(self.baseline_dataset)
