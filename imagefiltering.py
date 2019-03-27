import cv2
import numpy as np
from matplotlib import pyplot as plt

# img = cv2.imread('/Users/juanzapata/opencv/samples/data/sudoku.png',0)
img = cv2.imread('/home/juanzapata/OpenCV/samples/data/pic2.png',0)
# rows,cols,ch = img.shape

# CONVOLUTION
kernel = np.ones((5,5),np.float32)/25
dst= cv2.filter2D(img,-1,kernel)

# AVERAGING BLURED
# This is done by convolving image with a normalized box filter. It simply takes the average of all the pixels under kernel area and replace the central element. This is done by the function cv2.blur() or cv2.boxFilter(). Check the docs for more details about the kernel. We should specify the width and height of kernel. A 3x3 normalized box filter would look like np.ones((3,3),np.float32)/9
# If you dont want to use normalized box filter, use cv2.boxFilter(). Pass an argument normalize=False to the function.


blur = cv2.blur(img,(5,5))

# AVERAGING GAUSSIAN BLURED
# In this, instead of box filter, gaussian kernel is used. It is done with the function, cv2.GaussianBlur(). We should specify the width and height of kernel which should be positive and odd. We also should specify the standard deviation in X and Y direction, sigmaX and sigmaY respectively. If only sigmaX is specified, sigmaY is taken as same as sigmaX. If both are given as zeros, they are calculated from kernel size. Gaussian blurring is highly effective in removing gaussian noise from the image.

gblur = cv2.GaussianBlur(img,(5,5),0)


# MEDIAN BLURED
mblur = cv2.medianBlur(img,5)
# Here, the function cv2.medianBlur() takes median of all the pixels under kernel area and central element is replaced with this median value. This is highly effective against salt-and-pepper noise in the images. Interesting thing is that, in the above filters, central element is a newly calculated value which may be a pixel value in the image or a new value. But in median blurring, central element is always replaced by some pixel value in the image. It reduces the noise effectively. Its kernel size should be a positive odd integer.


# BILATERAL FILTER
blblur = cv2.bilateralFilter(img,9,75,75)
# cv2.bilateralFilter() is highly effective in noise removal while keeping edges sharp. But the operation is slower compared to other filters. We already saw that gaussian filter takes the a neighbourhood around the pixel and find its gaussian weighted average. This gaussian filter is a function of space alone, that is, nearby pixels are considered while filtering. It doesnt consider whether pixels have almost same intensity. It doesnt consider whether pixel is an edge pixel or not. So it blurs the edges also, which we dont want to do.

# Bilateral filter also takes a gaussian filter in space, but one more gaussian filter which is a function of pixel difference. Gaussian function of space make sure only nearby pixels are considered for blurring while gaussian function of intensity difference make sure only those pixels with similar intensity to central pixel is considered for blurring. So it preserves the edges since pixels at edges will have large intensity variation.

plt.subplot(321),plt.imshow(img),plt.title('Input')
plt.xticks([]), plt.yticks([])
plt.subplot(322),plt.imshow(dst),plt.title('Convolved Output')
plt.xticks([]), plt.yticks([])
plt.subplot(323),plt.imshow(blur),plt.title('Blured Output')
plt.xticks([]), plt.yticks([])
plt.subplot(324),plt.imshow(gblur),plt.title('Gaussian Blured Output')
plt.xticks([]), plt.yticks([])
plt.subplot(325),plt.imshow(mblur),plt.title('Median Blured Output')
plt.xticks([]), plt.yticks([])
plt.subplot(326),plt.imshow(blblur),plt.title('Bilateral Blured Output')
plt.xticks([]), plt.yticks([])
plt.show()
