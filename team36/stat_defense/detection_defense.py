from . import detection_statistics as ds
import torch

def detectAdversarial(X):
    PCA = ds.calculateStats(X)

    split_val = round(len(PCA) / 2)
    first_half = torch.sum(PCA[0:split_val])
    second_half = torch.sum(PCA[split_val:-1])

    if first_half >= second_half:
        # no adversarial attack detected
        return 0
    else:
        # high probability of adversarial attack
        return 1
