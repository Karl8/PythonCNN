import cPickle
import matplotlib.pyplot as plt

filename1 = "init_std=0.1"
filename2 = "lr=0.01"
'''
with open(filename1+'.data', "r") as f:
    train_loss_list_1=cPickle.load(f)
    train_acc_list_1=cPickle.load(f)
    test_loss_list_1=cPickle.load(f)
    test_acc_list_1=cPickle.load(f)

with open(filename2+'.data', "r") as f:
    train_loss_list_2=cPickle.load(f)
    train_acc_list_2=cPickle.load(f)
    test_loss_list_2=cPickle.load(f)
    test_acc_list_2=cPickle.load(f)

name = filename1 + " vs " + filename2

x = range(1, len(train_loss_list_1) + 1)
plt.title(name + 'Train Loss')
plt.ylim((0, 0.8))
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(x, train_loss_list_1, 'r', label=filename1)
plt.plot(x, train_loss_list_2, 'b', label=filename2)
plt.legend(loc='upper right')
plt.savefig(filename1 + "vs" + filename2+'_loss.png')

plt.clf()
plt.title(name + 'Train Accuracy')
plt.ylim((0.5, 1))
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(x, train_acc_list_1, 'r', label=filename1)
plt.plot(x, train_acc_list_2, 'b', label=filename2)
plt.legend(loc='lower right')
plt.savefig(filename1 + "vs" + filename2+'_acc.png')
'''

filename = 'lr=0.01_2'
with open(filename2+'.data', "r") as f:
    train_loss_list=cPickle.load(f)
    train_acc_list=cPickle.load(f)
    test_loss_list=cPickle.load(f)
    test_acc_list=cPickle.load(f)

x = range(1, len(train_loss_list) + 1)
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