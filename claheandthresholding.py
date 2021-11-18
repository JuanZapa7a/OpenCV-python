
# https://towardsdatascience.com/image-segmentation-using-pythons-scikit-image-module-533a61ecc980

#

import cv2
import numpy as np
from matplotlib import pyplot as plt
import argparse
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default="images/210908 10 H273.tif",
	help="path to input image")
args = vars(ap.parse_args())

img = cv2.imread(args["image"], 0)
equ = cv2.equalizeHist(img)
# cropping
rows, cols = img.shape
img = img[0:rows-70,0:cols]

# 0riginal image
f1 = plt.figure(1)
plt.subplot(3,2,1)
plt.title('Original Image')
plt.imshow(img, cmap='gray')
# plot histogram
plt.subplot(3,2,2)
plt.title('Original Image Hist.')
plt.hist(img.flat, bins=100, range=(0, 255))

# Blur
blur = cv2.GaussianBlur(img,(3,3),0)
plt.subplot(3,2,3)
plt.title('Blur Image')
plt.imshow(blur, cmap='gray')
# plot histogram
plt.subplot(3,2,4)
plt.title('Blur Image Hist.')
plt.hist(blur.flat, bins=100, range=(0, 255))

# Equ image
equ = cv2.equalizeHist(img)
plt.subplot(3,2,5)
plt.title('Equ Image')
plt.imshow(equ, cmap='gray')
# plot histogram
plt.subplot(3,2,6)
plt.title('Equ Image Hist.')
plt.hist(equ.flat, bins=100, range=(0, 255))
plt.xticks([]), plt.yticks([])

# for i in range(6):
#     plt.subplot(3, 2, i+1), plt.imshow(images[i], 'gray')
#     plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])
#
# plt.show()

f1.show()
