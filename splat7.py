# import cv2
# import numpy as np

# img = cv2.imread('images/7-5%_1750_centro_1.jpg')
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(img,50,150,apertureSize = 3)

# lines = cv2.HoughLines(edges,1,np.pi/180,200)
# for rho,theta in lines[0]:
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a*rho
#     y0 = b*rho
#     x1 = int(x0 + 1000*(-b))
#     y1 = int(y0 + 1000*(a))
#     x2 = int(x0 - 1000*(-b))
#     y2 = int(y0 - 1000*(a))

#     cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

# cv2.imwrite('houghlines3.jpg',img)


# Author: Juan Zapata (Universidad Politécnica de Cartagena)

# Use python splat4 -i images/image.xxx -c x
# Default use python splat4 with M3C102.bmp locates in ./images


# import the necessary packages
import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default="images/7-5%_1750_centro_1.bmp",
	help="path to input image")
# ap.add_argument("-c", "--connectivity", type=int, default=8, 
#     help="connectivity for connected component analysis")
args = vars(ap.parse_args())

# START:

# Reading
img = cv2.imread(args["image"])
rows,cols,ch = img.shape
plt.title('Original Image')
plt.imshow(img)
plt.show()

# Cropping
img = img[0:rows-70,0:cols]
plt.title('Cropped Image')
plt.imshow(img)
plt.show()

# Gray
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
plt.title('Gray Image')
plt.imshow(img_gray, cmap='gray')
#plt.savefig('images/GrayImage.png')
plt.show()

# Blur
img_blur = cv2.medianBlur(img_gray, 5)
plt.title('Blur Image')
plt.imshow(img_blur, cmap='gray')
#plt.savefig('images/Blur Image.png')
plt.show()

# threshold
img_thr = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV|cv2.THRESH_BINARY, 11, 2)
#img_thr = cv2.threshold(img_blur, 0, 255,  cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
plt.title('Thresholding Image')
plt.imshow(img_thr, cmap='gray')
#plt.savefig('images/Thesholding Image.png')
plt.show()

# Canny
edge_image = cv2.Canny(img_blur, 0, 255 , apertureSize=3)
plt.title('Edge Image')
plt.imshow(edge_image, cmap='gray')
#plt.savefig('images/GrayImage.png')
plt.show()

# lines
lines = cv2.HoughLines(edge_image,1,np.pi/180,200)
for rho,theta in lines[0]:
	a = np.cos(theta)
	b = np.sin(theta)
	x0 = a*rho
	y0 = b*rho
	x1 = int(x0 + 1000*(-b))
	y1 = int(y0 + 1000*(a))
	x2 = int(x0 - 1000*(-b))
	y2 = int(y0 - 1000*(a))
	cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
plt.title('Original Image plus lines')
plt.imshow(img)
plt.show()

minLineLength = 10
maxLineGap = 10
lines = cv2.HoughLinesP(edge_image,1,np.pi/180,50,minLineLength,maxLineGap)
for x1,y1,x2,y2 in lines[0]:
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
plt.title('Original Image plus lines2')
plt.imshow(img)
plt.show()