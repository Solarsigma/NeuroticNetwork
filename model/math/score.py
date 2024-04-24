import numpy as np

def score(y_pred, y_true, method='r2'):
    match method:
        case 'r2':
            ssr = np.sum((y_pred - y_true)**2)
            sst = np.sum((y_pred - np.mean(y_true))**2)
            return 1 - ssr/sst