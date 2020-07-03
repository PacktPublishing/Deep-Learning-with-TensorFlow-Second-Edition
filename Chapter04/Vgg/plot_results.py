#Import the libraries needed
import numpy as np
import scipy.io
import scipy.misc
import tensorflow as tf 
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

plt.figure(figsize=(15,15))
for i in range(11):
    plt.subplot(5,5,i+1)
    if i==0:
        imshow(plt.imread('out/%d.png' % i))
    else:
        imshow(plt.imread('out/%d00.png' % i))
    plt.axis('off')
    plt.title('iteration %d00' % i)
plt.show()
