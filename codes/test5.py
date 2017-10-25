import numpy as np
input = np.ones([3, 2])
target = np.array([[1, 0], [0, 1], [1, 0]])
print "input shape", input.shape
h = np.exp(input)
print "h shape", h.shape
s = np.sum(h, axis=1)

s = np.repeat(s, 2).reshape([3, 2])
print "s shape", s.shape
print s
h = h / s
print h
e = np.mean(-np.sum(target * np.log(h), axis=1))
print e 
