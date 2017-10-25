from __future__ import division
import numpy as np


class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        return 0.5 * np.mean(np.sum(np.square(input - target), axis=1))

    def backward(self, input, target):
        return (input - target) / len(input)


class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        '''Your codes here'''
        h = np.exp(input)
        s = np.sum(h, axis=1)
        s = np.repeat(s, input.shape[1]).reshape(input.shape)
        h = h / s
        self._saved_h = h
        e = np.mean(-np.sum(target * np.log(h), axis=1))    
        return e

    def backward(self, input, target):
        '''Your codes here'''
        h = self._saved_h
        return (h - target) / len(input)