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
        # std = self.forward_std(input, target)
        # if e == std:
        #     print "loss good"
        # else:
        #     print "!!!!!!!!!!! loss bad"
        return e

    def backward(self, input, target):
        '''Your codes here'''
        h = self._saved_h
        # std = self.backward_std(input, target)
        # out = (h - target) / len(input)
        # if out.all() == std.all():
        #     print "loss good"
        # else:
        #     print "!!!!!!!!!!! loss bad"
        return (h - target) / len(input)
    
    def forward_std(self, input, target):
        input_exp = np.exp(input)
        h = np.divide(input_exp.T, np.sum(input_exp, 1)).T
        self._saved_h = h
        return np.mean(-np.sum(np.multiply(np.log(h), target), 1))

    def backward_std(self, input, target):
        h = self._saved_h
        return (h - target) / target.shape[0]
