from utils import LOG_INFO, onehot_encoding, calculate_acc
import numpy as np
import os

def data_iterator(x, y, batch_size, shuffle=True):
    indx = range(len(x))
    if shuffle:
        np.random.shuffle(indx)

    for start_idx in range(0, len(x), batch_size):
        end_idx = min(start_idx + batch_size, len(x))
        yield x[indx[start_idx: end_idx]], y[indx[start_idx: end_idx]]


def train_net(model, loss, config, inputs, labels, batch_size, disp_freq):

    iter_counter = 0
    loss_list = []
    acc_list = []

    train_loss_list = []
    train_acc_list = []

    for input, label in data_iterator(inputs, labels, batch_size):
        target = onehot_encoding(label, 10)
        iter_counter += 1

        # forward net
        output = model.forward(input)
        # calculate loss
        loss_value = loss.forward(output, target)
        # generate gradient w.r.t loss
        grad = loss.backward(output, target)
        # backward gradient

        model.backward(grad)
        # update layers' weights
        model.update(config)
        #os.system("pause")
        acc_value = calculate_acc(output, label)
        loss_list.append(loss_value)
        train_loss_list.append(loss_value)
        acc_list.append(acc_value)
        train_acc_list.append(acc_value)

        if iter_counter % disp_freq == 0:
            msg = '  Training iter %d, batch loss %.4f, batch acc %.4f' % (iter_counter, np.mean(loss_list), np.mean(acc_list))
            loss_list = []
            acc_list = []
            LOG_INFO(msg)
    return np.mean(train_loss_list), np.mean(train_acc_list)

def test_net(model, loss, inputs, labels, batch_size):
    loss_list = []
    acc_list = []

    test_loss_list = []
    test_acc_list = []

    for input, label in data_iterator(inputs, labels, batch_size, shuffle=False):
        target = onehot_encoding(label, 10)
        output = model.forward(input)
        loss_value = loss.forward(output, target)
        acc_value = calculate_acc(output, label)
        loss_list.append(loss_value)
        test_loss_list.append(loss_value)
        acc_list.append(acc_value)
        test_acc_list.append(acc_value)

    msg = '    Testing, total mean loss %.5f, total acc %.5f' % (np.mean(loss_list), np.mean(acc_list))
    LOG_INFO(msg)

    return np.mean(test_loss_list), np.mean(test_acc_list)
