import detection_statistics
import numpy as np

def detectAdversarial(X)
    PCA = detection_statistics.calculateStats(X)

    split_val = round(len(PCA) / 2)

    first_half = np.sum(PCA[:split_val])
    second_half = np.sum(PCA[split_val:])

    if first_half >= second_half:
        # no adversarial attack detected
        return 0
    else:
        # high probability of adversarial attack
        return 1
