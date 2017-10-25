import numpy as np
import matplotlib.pyplot as plt
import cPickle
def vis_square(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
    
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())
    #print data
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    print data
    plt.imshow(data); plt.axis('off')
img = []
with open("img.data", "r") as f:
    img = cPickle.load(f)

n, c, h, w = img.shape
print img[0].shape
img = img[0:4,:,:,:].reshape([-1, h, w])
vis_square(img)
plt.savefig('digit.png')
#plt.imshow(img[0])
#plt.show()