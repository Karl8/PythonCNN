from im2col import data_to_2d, filter_to_2d, filter2d_to_orig
import numpy as np
a = np.arange(64).reshape([2, 2, 4, 4])
b = np.arange(54).reshape([3, 2, 3, 3])

print a
print b
c = data_to_2d(a, b, 1, 1, 1, 1)
d = filter_to_2d(b)
print c.T.shape
print c.T
print d

e = np.dot(d, c)
print e.shape
print e

f = filter2d_to_orig(e, [4, 4])
g = f.transpose((1, 0, 2, 3))
print f.shape
print f
print g.shape
print g