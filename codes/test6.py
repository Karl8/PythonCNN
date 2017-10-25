import numpy as np
from im2col import data_to_2d
input = np.arange(96).reshape([2, 3, 4, 4])
kernel_size = 2
pad = 0
n, c_in, h_in, w_in = input.shape
h_out = (h_in + 2 * pad) / kernel_size
w_out = (w_in + 2 * pad) / kernel_size
    
W = np.ones([c_in * c_in, 1, kernel_size, kernel_size]) / np.square(kernel_size)
input_mat = data_to_2d(input.reshape([n * c_in, 1, h_in, w_in]), W, kernel_size, kernel_size, pad, pad)
print input_mat
output = np.mean(input_mat, axis=0).reshape(n, c_in, h_out, w_out)
print input
print input_mat
print output