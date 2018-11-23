# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 19:16:05 2013
@author: duan
"""

import numpy as np
from matplotlib import pyplot as plt
import cv2

# Load an color image in grayscale
img = cv2.imread('messi5.jpg')
print img.shape
print img.size
# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# img = img[:,:,::-1]
# img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# plt.imshow(img, interpolation = 'bicubic')
# plt.xticks([]), plt.yticks([]) # to hide tick values on X and Y axis
# plt.show()
cv2.filter2D()