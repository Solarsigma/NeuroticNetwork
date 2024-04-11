import numpy as np
from enum import Enum

def _sigmoid_(l):
    return 1 / (1 + np.exp(-l))

def _tanh_d_(l):
    return 1 - np.tanh(l)**2

def _softmax_(l):
    return np.exp(l) / np.sum(np.exp(l), axis=0)

def _softmax_d_(l):
    s = _softmax_(l)
    return diagflat(s) - self_product(s)

def diagflat(arr):
    if len(arr.shape) == 2:
        return np.dstack([np.diagflat(col) for col in arr.T])
    return np.diagflat(arr)

def self_product(arr):
    return np.dstack([np.matmul(np.atleast_2d(col).T, np.atleast_2d(col)) for col in arr.T])

class CustomEnum(Enum):
    @classmethod
    def get_by_value(cls, value):
        for enum_member in list(cls):
            if enum_member.value == value:
                return enum_member
        return None