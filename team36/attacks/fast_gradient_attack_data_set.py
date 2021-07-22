import torch


# source: https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        batch = []
        for j in range(i, i + n):
            batch.append(lst[j])
        yield batch


# "fast gradient sign method" from EXPLAINING AND HARNESSING ADVERSARIAL EXAMPLES
# gradient calculation from
#   https://stackoverflow.com/questions/54754153/autograd-grad-for-tensor-in-pytorch
class FastSignGradientAttackDataSet(torch.utils.data.Dataset):
    def __init__(self, baseline_dataset, model, criterion, epsilon):
        self.baseline_dataset = baseline_dataset
        self.model = model
        self.criterion = criterion
        self.epsilon = epsilon

        self.attack_dataset = self.__build_attack_dataset__()

    def __build_attack_dataset__(self):
        attack_dataset = []
        batch_size = 128
        j = 0
        for batch in chunks(self.baseline_dataset, batch_size):
            j += 1
            print(j, len(self.baseline_dataset) // batch_size)
            inputs = torch.zeros((len(batch),) + batch[0][0].shape)
            targets = torch.zeros((len(batch)), dtype=torch.long)
            for i, x in enumerate(batch):
                inputs[i] = x[0]
                targets[i] = x[1]
            inputs.requires_grad_()

            out = self.model(inputs)
            loss = self.criterion(out, targets)
            loss_gradient = torch.autograd.grad(outputs=loss, inputs=inputs)

            epsilon = 0.25
            eta = epsilon * torch.sign(loss_gradient[0])
            adversarial_inputs = inputs + eta

            for index in range(adversarial_inputs.shape[0]):
                adversarial_input = adversarial_inputs[index]
                min = torch.min(adversarial_input)
                max = torch.max(adversarial_input)
                adversarial_input = (adversarial_input - min) / (max - min)
                attack_dataset.append((adversarial_input, targets[i].item()))

        return attack_dataset

    def __getitem__(self, index):
        return self.attack_dataset[index]

    def __len__(self):
        return len(self.attack_dataset)
