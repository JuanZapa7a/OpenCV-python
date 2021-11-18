import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default="images/210908 10 H273.tif",
	help="path to input image")
args = vars(ap.parse_args())

# Reading
img = cv2.imread(args["image"])
rows, cols, ch = img.shape
plt.title('Original Image')
plt.imshow(img)
plt.show()

# Cropping
img = img[0:rows-70, 0:cols]
plt.title('Cropped Image')
plt.imshow(img)
plt.show()

# Convert the image to gray-scale
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.title('Gray Image')
plt.imshow(img, cmap='gray')
#plt.savefig('images/GrayImage.png')
plt.show()

ret, thresh_hold = cv2.threshold(img,100,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
plt.title('Edge Image (binary thresh)')
plt.imshow(thresh_hold, cmap='gray')
#plt.savefig('images/GrayImage.png')
plt.show()

ret, thresh_hold1 = cv2.threshold(img,100,255,cv2.THRESH_BINARY_INV)
plt.title('Edge Image (binary inv. thresh)')
plt.imshow(thresh_hold1, cmap='gray')
#plt.savefig('images/GrayImage.png')
plt.show()

ret, thresh_hold2 = cv2.threshold(img,100,255,cv2.THRESH_TOZERO)
plt.title('Edge Image (to zero thresh)')
plt.imshow(thresh_hold2, cmap='gray')
#plt.savefig('images/GrayImage.png')
plt.show()

ret, thresh_hold3 = cv2.threshold(img,100,255,cv2.THRESH_TOZERO_INV)
plt.title('Edge Image (to zero inv. thresh)')
plt.imshow(thresh_hold3, cmap='gray')
#plt.savefig('images/GrayImage.png')
plt.show()

ret, thresh_hold4 = cv2.threshold(img,100,255,cv2.THRESH_TRUNC)
plt.title('Edge Image (trunc. thresh)')
plt.imshow(thresh_hold4, cmap='gray')
#plt.savefig('images/GrayImage.png')
plt.show()

ret, thresh_hold5 = cv2.threshold(img,100,255,cv2.THRESH_OTSU)
plt.title('Edge Image (Otsu thresh)')
plt.imshow(thresh_hold5, cmap='gray')
#plt.savefig('images/GrayImage.png')
plt.show()

blur = cv2.GaussianBlur(img,(1,1),0)
ret, thresh_hold = cv2.threshold(blur,100,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
plt.title('Edge Image (binary thresh)')
plt.imshow(thresh_hold, cmap='gray')
#plt.savefig('images/GrayImage.png')
plt.show()

#thresh_hold = cv2.resize(thresh_hold, (960, 540))
#cv2.imshow('Binary Threshold Image', thresh_hold)