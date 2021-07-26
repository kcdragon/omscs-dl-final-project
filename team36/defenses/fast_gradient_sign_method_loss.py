import torch
import torch.nn as nn


# source: Goodfellow et al, Explaining and harnessing adversarial examples
class FastGradientSignMethodLoss(nn.Module):
    def __init__(self, model, criterion, alpha=0.5, epsilon=0.25):
        super().__init__()

        self.model = model
        self.criterion = criterion
        self.alpha = alpha
        self.epsilon = epsilon

    # J'(theta, x, y) = alpha * J(theta, x, y) +
    #                   (1 − alpha) * J(theta, x + epsilon * sign(gradientJ(theta, x, y)).
    def forward(self, input, target):
        # alpha * J(theta, x, y)
        loss = self.criterion(input, target)
        loss.backward(retain_graph=True)  # retain the graph since we are calling loss again later

        # (1 − alpha) * J(theta, x + epsilon * sign(gradientJ(theta, x, y))
        loss_gradient = torch.autograd.grad(outputs=loss, inputs=input, retain_graph=True)
        eta = self.epsilon * loss_gradient[0].sign()
        fgsm_loss = self.criterion(input + eta, target)

        return self.alpha * loss + (1 - self.alpha) * fgsm_loss
