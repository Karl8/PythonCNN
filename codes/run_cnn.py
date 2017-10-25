from network import Network
from layers import Relu, Linear, Conv2D, AvgPool2D, Reshape
from utils import LOG_INFO
from loss import EuclideanLoss, SoftmaxCrossEntropyLoss
from solve_net import train_net, test_net
from load_data import load_mnist_4d
import time
import cPickle
import json
import matplotlib.pyplot as plt

train_data, test_data, train_label, test_label = load_mnist_4d('data')

# Your model defintion here
# You should explore different model architecture
model = Network()
model.add(Conv2D('conv1', 1, 4, 3, 1, 1)) # c_in, c_out, k, pad, init_std, output shape: N x 4 X 28 x 28
model.add(Relu('relu1'))
model.add(AvgPool2D('pool1', 2, 0))  # output shape: N x 4 x 14 x 14
model.add(Conv2D('conv2', 4, 4, 3, 1, 1))
model.add(Relu('relu2'))
model.add(AvgPool2D('pool2', 2, 0))  # output shape: N x 4 x 7 x 7
model.add(Reshape('flatten', (-1, 196)))
model.add(Linear('fc3', 196, 10, 0.1))
loss = SoftmaxCrossEntropyLoss(name='softmaxloss')
#loss = EuclideanLoss(name="euclidanloss")
# Training configuration
# You should adjust these hyperparameters
# NOTE: one iteration means model forward-backwards one batch of samples.
#       one epoch means model has gone through all the training samples.
#       'disp_freq' denotes number of iterations in one epoch to display information.

config = {
    'learning_rate': 0.01,
    'weight_decay': 0,
    'momentum': 0.9,
    'batch_size': 100,
    'max_epoch': 100,
    'disp_freq': 100,
    'test_epoch': 1
}

train_acc_list = []
train_loss_list = []
test_acc_list = []
test_loss_list = []
start_time = time.time()

for epoch in range(config['max_epoch']):
    LOG_INFO('Training @ %d epoch...' % (epoch))
    train_loss, train_acc = train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'])
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    if epoch % config['test_epoch'] == 0:
        LOG_INFO('Testing @ %d epoch...' % (epoch))
        test_loss, test_acc = test_net(model, loss, train_data, train_label, config['batch_size'])
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)

during = time.time() - start_time;

model.show()

filename = "../result/" + "kernel=5"
with open(filename +".txt", 'w') as f:
    json.dump(model.info, f)
    json.dump(config, f)
    json.dump([test_acc_list[-1], during], f)

with open(filename +".data", 'w') as f:
    cPickle.dump(train_loss_list, f)
    cPickle.dump(train_acc_list, f)
    cPickle.dump(test_loss_list, f)
    cPickle.dump(test_acc_list, f)
    

x = range(1, config['max_epoch'] + 1)
plt.title('Train/Test Loss')
plt.ylim((0, 0.8))
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(x, train_loss_list, 'r', label='Train')
plt.plot(x, test_loss_list, 'b', label='Test')
plt.legend(loc='upper right')
plt.savefig(filename+'_loss.png')

plt.clf()
plt.title('Train/Test Accuracy')
plt.ylim((0.5, 1))
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(x, train_acc_list, 'r', label='Train')
plt.plot(x, test_acc_list, 'b', label='Test')
plt.legend(loc='lower right')
plt.savefig(filename+'_acc.png')
