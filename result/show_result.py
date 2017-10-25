import cPickle
import matplotlib.pyplot as plt

filename1 = "init_std=0.1"
filename2 = "init_std=1"
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
plt.title(name + ' Train Loss')
plt.ylim((0, 0.8))
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(x, train_loss_list_1, 'r', label=filename1)
plt.plot(x, train_loss_list_2, 'b', label=filename2)
plt.legend(loc='upper right')
plt.savefig(filename1 + "vs" + filename2+'_loss.png')

plt.clf()
plt.title(name + ' Train Accuracy')
plt.ylim((0.5, 1))
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(x, train_acc_list_1, 'r', label=filename1)
plt.plot(x, train_acc_list_2, 'b', label=filename2)
plt.legend(loc='lower right')
plt.savefig(filename1 + "vs" + filename2+'_acc.png')

result = ("%.3f&%.3f&%.2f\\%%&%.2f\\%%\n%.3f&%.3f&%.2f\\%%&%.2f\\%%") % (train_loss_list_1[-1], test_loss_list_1[-1],train_acc_list_1[-1]*100, test_acc_list_1[-1]*100,train_loss_list_2[-1], test_loss_list_2[-1],train_acc_list_2[-1]*100, test_acc_list_2[-1]*100)
with open(filename1 + "vs" + filename2+'.txt', "w") as f:
    f.write(result)