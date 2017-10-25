import numpy as np
import os
from im2col import data_to_2d

def conv2d_forward(input, W, b, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        W: weight, shape = c_out (#output channel) x c_in (#input channel) x k (#kernel_size) x k (#kernel_size)
        b: bias, shape = c_out
        kernel_size: size of the convolving kernel (or filter)
        pad: number of zero added to both sides of input

    Returns:
        output: shape = n (#sample) x c_out (#output channel) x h_out x w_out,
            where h_out, w_out is the height and width of output, after convolution
    '''
    n, c_in, h_in, w_in = input.shape;
    c_out = b.shape[0]
    h_out = h_in + 2 * pad - kernel_size + 1
    w_out = w_in + 2 * pad - kernel_size + 1

    input_mat = data_to_2d(input, W, 1, 1, pad, pad)
    W_mat = W.reshape(c_out, c_in * kernel_size * kernel_size)
    output_mat = np.dot(W_mat, input_mat)
    output_mat = output_mat + b.reshape(-1, 1)
    output_t = output_mat.reshape(c_out, n, h_out, w_out)
    output = output_t.transpose((1, 0, 2, 3))
    return output

def conv2d_backward(input, grad_output, W, b, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        grad_output: shape = n (#sample) x c_out (#output channel) x h_out x w_out
        W: weight, shape = c_out (#output channel) x c_in (#input channel) x k (#kernel_size) x k (#kernel_size)
        b: bias, shape = c_out
        kernel_size: size of the convolving kernel (or filter)
        pad: number of zero added to both sides of input

    Returns:
        grad_input: gradient of input, shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        grad_W: gradient of W, shape = c_out (#output channel) x c_in (#input channel) x k (#kernel_size) x k (#kernel_size)
        grad_b: gradient of b, shape = c_out
    '''
    n, c_in, h_in, w_in = input.shape
    n, c_out, h_out, w_out = grad_output.shape
    pad_new = kernel_size - 1

    W_rot180 = W[:,:,::-1,::-1]
    grad_output_mat = data_to_2d(grad_output, W_rot180, 1, 1, pad_new, pad_new)
    W_rot180_t = W_rot180.transpose((1, 0, 2, 3))
    W_rot180_t_mat = W_rot180_t.reshape(c_in, c_out * kernel_size * kernel_size)
    grad_input_pad_t_mat = np.dot(W_rot180_t_mat, grad_output_mat)
    grad_input_pad_t = grad_input_pad_t_mat.reshape(c_in, n, h_in + pad * 2, w_in + pad * 2)
    grad_input_pad = grad_input_pad_t.transpose((1, 0, 2, 3))
    if pad == 0:
        grad_input = grad_input_pad
    else:
        grad_input = grad_input_pad[:,:,pad:-pad, pad:-pad]

    input_t = input.transpose(1, 0, 2, 3)
    input_t_mat = data_to_2d(input_t, grad_output, 1, 1, pad, pad)
    grad_output_t = grad_output.transpose(1, 0, 2, 3)
    grad_output_t_mat = grad_output_t.reshape(c_out, n * h_out * w_out)
    grad_W_mat = np.dot(grad_output_t_mat, input_t_mat)
    grad_W = grad_W_mat.reshape(c_out, c_in, kernel_size, kernel_size)
    grad_b = np.sum(grad_output, axis=(0, 2, 3))
    return grad_input, grad_W, grad_b


def avgpool2d_forward(input, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        kernel_size: size of the window to take average over
        pad: number of zero added to both sides of input

    Returns:
        output: shape = n (#sample) x c_in (#input channel) x h_out x w_out,
            where h_out, w_out is the height and width of output, after average pooling over input
    '''
    n, c_in, h_in, w_in = input.shape
    h_out = (h_in + 2 * pad) / kernel_size
    w_out = (w_in + 2 * pad) / kernel_size
    
    W = np.ones([c_in * c_in, 1, kernel_size, kernel_size]) / np.square(kernel_size)
    input_mat = data_to_2d(input.reshape([n * c_in, 1, h_in, w_in]), W, kernel_size, kernel_size, pad, pad)
    output = np.mean(input_mat, axis=0).reshape(n, c_in, h_out, w_out)
    return output


def avgpool2d_backward(input, grad_output, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        grad_output: shape = n (#sample) x c_in (#input channel) x h_out x w_out
        kernel_size: size of the window to take average over
        pad: number of zero added to both sides of input

    Returns:
        grad_input: gradient of input, shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
    '''
    kron = np.kron(grad_output, np.ones([kernel_size, kernel_size])) / np.square(kernel_size)
    if pad == 0:
        grad_input = kron
    else:
        grad_input = kron[:,:,pad:-pad, pad:-pad]
    return grad_input