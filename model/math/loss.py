import numpy as np
from . import util

class Loss(util.CustomEnum):
    MSE = 'mse'
    HUBER = 'huber'
    BINARY_CE = 'binary-cross-entropy'
    CATEGORICAL_CE = 'categorical-cross-entropy'
    MAE = 'mae'

    def __init__(self, value):
        self._value_ = value
        self.params = {}

    def set_params(self, **hyperparams):
        self.params = hyperparams
        return self

    def compute(self, y_true, y_pred):
        match self:
            case Loss.MSE:
                return np.sum((y_pred - y_true)**2, axis=0)
            case Loss.HUBER:
                delta = self.params.get('delta', 1)
                return np.sum(np.where(np.abs(y_pred - y_true) < delta, ((y_pred - y_true)**2)/2, delta*(y_pred - y_true - delta/2)), axis=0)
            case Loss.BINARY_CE:
                return np.sum(-(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)), axis=0)
            case Loss.CATEGORICAL_CE:
                return -np.sum(y_true * np.log(y_pred), axis=0)
            case _:
                return np.sum(y_pred - y_true, axis=0)
    
    def gradient(self, y_true, y_pred):
        match self:
            case Loss.MSE:
                return 2*(y_pred - y_true)
            case Loss.HUBER:
                delta = self.params.get('delta', 1)
                return np.where(np.abs(y_pred - y_true) < delta, (y_pred - y_true), delta*np.ones_like(y_pred))
            case Loss.BINARY_CE:
                return -(y_true / y_pred - (1 - y_true) / (1 - y_pred))
            case Loss.CATEGORICAL_CE:
                return - y_true / y_pred
            case _:
                return np.ones_like(y_pred)
