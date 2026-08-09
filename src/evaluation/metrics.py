import numpy as np

def wape(target, prediction):
    target, prediction = np.asarray(target), np.asarray(prediction)
    return np.abs(target - prediction).sum() / np.abs(target).sum()