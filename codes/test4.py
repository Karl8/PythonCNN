from im2col import data_to_2d, filter_to_2d, filter2d_to_orig
import numpy as np
pad = 0
kernel_size = 3
input = np.arange(32).reshape([1, 2, 4, 4])
#input = np.ones([1, 2, 4, 4])
#grad_output = np.arange(588).reshape([1, 3, 14, 14])
grad_output = np.ones([1, 3, 14, 14])
n, c_in, h_in, w_in = input.shape
n, c_out, h_out, w_out = grad_output.shape
W = np.ones([3, 2, 3, 3]) / 4

kron = np.kron(grad_output, np.ones([kernel_size, kernel_size])) / np.square(kernel_size)
#print grad_output
print kron.shape
grad_input = kron
x = kron[:, :, 0:-0, 0:-0]
print grad_input.shape
print grad_input
print x.shape
#print grad_input