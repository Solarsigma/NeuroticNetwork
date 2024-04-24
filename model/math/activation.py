import numpy as np
from . import util

class Activation(util.CustomEnum):
    RELU = 'relu'
    LEAKY_RELU = 'lrelu'
    TANH = 'tanh'
    IDENTITY = 'identity'
    STEP = 'step'
    SIGMOID = 'sigmoid'
    GELU = 'gelu'
    SWISH = 'swish'
    ELU = 'elu'
    SOFTPLUS = 'softplus'
    SOFTMAX = 'softmax'

    def __init__(self, value):
        self._value_ = value
        self.params = {}

    def set_params(self, **hyperparams):
        self.params = hyperparams
        return self

    def compute(self, l):
        match self:
            case Activation.RELU:
                return np.maximum(0, l)
            case Activation.RELU:
                return np.maximum(0, l)
            case Activation.TANH:
                return np.tanh(l)
            case Activation.LEAKY_RELU:
                alpha = self.params.get('alpha', 0.01)
                return np.maximum(alpha * l, l)
            case Activation.STEP:
                return np.where(l > 0, 1, 0)
            case Activation.SIGMOID:
                return util._sigmoid_(l)
            case Activation.GELU:
                return 0.5*l * (1 + np.tanh(np.sqrt(2/np.pi) * (l + 0.044715*(l**3))))
            case Activation.SWISH:
                beta = self.params.get('beta', 1)
                return l * util._sigmoid_(beta * l)
            case Activation.ELU:
                alpha = self.params.get('alpha', 1)
                return np.where(l > 0, l, alpha*(np.exp(l) - 1))
            case Activation.SOFTPLUS:
                return np.log(1 + np.exp(l))
            case Activation.SOFTMAX:
                return util._softmax_(l)
            case _:
                return l
    
    def gradient(self, l):
        match self:
            case Activation.RELU:
                return util.diagflat(np.heaviside(l, 0))
            case Activation.TANH:
                return util.diagflat(util._tanh_d_(l))
            case Activation.LEAKY_RELU:
                alpha = self.params.get('alpha', 0.01)
                return util.diagflat(np.where(l < 0, alpha, 1))
            case Activation.STEP:
                return util.diagflat(np.zeros_like(l))
            case Activation.SIGMOID:
                return util.diagflat(util._sigmoid_(l) * (1 - util._sigmoid_(l)))
            case Activation.GELU:
                return util.diagflat(0.5*l * (1 + (util._tanh_d_(np.sqrt(2/np.pi) * (l + 0.044715*(l**3))) * np.sqrt(2/np.pi) * (1 + 3*0.044715*(l**2)))))
            case Activation.SWISH:
                beta = self.params.get('beta', 1)
                return util.diagflat(util._sigmoid_(beta*l) + beta*l * util._sigmoid_(beta*l) * (1 - util._sigmoid_(beta*l)))
            case Activation.ELU:
                alpha = self.params.get('alpha', 1)
                return util.diagflat(np.where(l > 0, 1, alpha*np.exp(l)))
            case Activation.SOFTPLUS:
                return util.diagflat(np.exp(l) / (1 + np.exp(l)))
            case Activation.SOFTMAX:
                return util._softmax_d_(l)
            case _:
                return util.diagflat(np.ones_like(l))
