from im2col import data_to_2d, filter_to_2d, filter2d_to_orig
import numpy as np
pad = 1
kernel_size = 3
input = np.arange(32).reshape([1, 2, 4, 4])
#input = np.ones([1, 2, 4, 4])
#grad_output = np.arange(48).reshape([1, 3, 4, 4])
grad_output = np.ones([1, 3, 4, 4])
n, c_in, h_in, w_in = input.shape
n, c_out, h_out, w_out = grad_output.shape
#W = np.ones([3, 2, 3, 3]) / 4
W = np.arange(54).reshape([3, 2, 3, 3])
W_rot180 = W[:,:,::-1,::-1]
pad_new = kernel_size - pad - 1
grad_output_mat = data_to_2d(grad_output, W_rot180, 1, 1, pad_new, pad_new)
W_rot180_t = W_rot180.transpose((1, 0, 2, 3))
W_rot180_t_mat = filter_to_2d(W_rot180_t)
print grad_output
print "grad_output_mat", grad_output_mat.shape
print grad_output_mat
print "W_rot180_t", W_rot180_t.shape
print W_rot180_t
print "W_rot180_t_mat", W_rot180_t_mat.shape
print W_rot180_t_mat
grad_input_t_mat = np.dot(W_rot180_t_mat, grad_output_mat)
grad_input_t = filter2d_to_orig(grad_input_t_mat, [h_in, w_in])
grad_input = grad_input_t.transpose((1, 0, 2, 3))
print grad_input_t_mat.shape
print grad_input_t_mat
print grad_input_t.shape
print grad_input.shape
print grad_input
'''
grad_input: gradient of input, shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        grad_W: gradient of W, shape = n (#sample) x c_out (#output channel) x h_out x w_out
        grad_b: gradient of b, shape = c_out
'''
input_t = input.transpose(1, 0, 2, 3)
input_t_mat = data_to_2d(input_t, grad_output, 1, 1, pad, pad)
print "input_t_mat", input_t_mat.shape
print input_t_mat
grad_output_t = grad_output.transpose(1, 0, 2, 3)

grad_output_t_mat = filter_to_2d(grad_output_t)
print "grad_output_t_mat", grad_output_t_mat.shape

grad_W_mat = np.dot(grad_output_t_mat, input_t_mat)
grad_W = filter2d_to_orig(grad_W_mat, [kernel_size, kernel_size])
print "grad_W_mat", grad_W_mat.shape
print "grad_W", grad_W.shape
print grad_W

grad_b = np.sum(grad_output, axis=(0, 2, 3))
print "grad_b", grad_b.shape
print grad_b