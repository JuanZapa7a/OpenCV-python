import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('j.png', 0)
kernel = np.ones((5, 5), np.uint8)
# We manually created a structuring elements with help of Numpy. It is rectangular shape. But in some cases, you may need elliptical/circular shaped kernels. So for this purpose, OpenCV has a function, cv2.getStructuringElement(). You just pass the shape and size of the kernel, you get the desired kernel.
#   Rectangular Kernel  cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
#   Elliptical Kernel   cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
#   Cross-shaped Kernel cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))

erosion = cv2.erode(img, kernel, iterations = 1)
dilation = cv2.dilate(img, kernel, iterations = 1)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

plt.subplot(421), plt.imshow(img), plt.title('Input')
plt.xticks([]), plt.yticks([])
plt.subplot(422), plt.imshow(erosion), plt.title('Erosion')
plt.xticks([]), plt.yticks([])
plt.subplot(423), plt.imshow(dilation), plt.title('dilation')
plt.xticks([]), plt.yticks([])
plt.subplot(424), plt.imshow(opening), plt.title('opening')
plt.xticks([]), plt.yticks([])
plt.subplot(425), plt.imshow(closing), plt.title('closing')
plt.xticks([]), plt.yticks([])
plt.subplot(426), plt.imshow(gradient), plt.title('gradient')
plt.xticks([]), plt.yticks([])
plt.subplot(427), plt.imshow(tophat), plt.title('tophat')
plt.xticks([]), plt.yticks([])
plt.subplot(428), plt.imshow(blackhat), plt.title('blackhat')
plt.xticks([]), plt.yticks([])
# plt.subplot(429),plt.imshow(blblur),plt.title('Bilateral Blured Output')
# plt.xticks([]), plt.yticks([])
plt.show()
