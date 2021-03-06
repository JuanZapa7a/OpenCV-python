import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('/Users/juanzapata/MEGA/Downloads/opencv/samples/data/sudoku.png',0)
#img = cv.imread('/home/juanzapata/OpenCV/samples/data/sudoku.jpg',0)
#plt.imshow(img)
img = cv2.medianBlur(img,5)

ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
#plt.imshow(img,cmap='gray')

# In this, the algorithm calculate the threshold for a small regions of the image. So we get different thresholds for different regions of the same image and it gives us better results for images with varying illumination.
# Adaptive Method - It decides how thresholding value is calculated.
#         cv.ADAPTIVE_THRESH_MEAN_C : threshold value is the mean of neighbourhood area.
#         cv.ADAPTIVE_THRESH_GAUSSIAN_C : threshold value is the weighted sum of neighbourhood values where weights are a gaussian window.
# Block Size - It decides the size of neighbourhood area.
#
# C - It is just a constant which is subtracted from the mean or weighted mean calculated.


th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]

# Python 2 code that uses xrange(...) unchanged, and any
# range(...) replaced with list(range(...))

for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()
