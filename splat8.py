# import the necessary packages
import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default="images/lanes.jpg",
	help="path to input image")
# ap.add_argument("-c", "--connectivity", type=int, default=8,
#     help="connectivity for connected component analysis")
args = vars(ap.parse_args())


# Read image
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

# Convert the image to gray-scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.title('Gray Image')
plt.imshow(gray, cmap='gray')
#plt.savefig('images/GrayImage.png')
plt.show()

# Find the edges in the image using canny detector
edges = cv2.Canny(gray, 50, 200)
plt.title('Edge Image')
plt.imshow(edges, cmap='gray')
#plt.savefig('images/GrayImage.png')
plt.show()

# Detect points that form a line
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=10, maxLineGap=250)
# Draw lines on the image
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
# Show result
#cv2.imshow("Result Image", img)
plt.title('Original Image plus lines')
plt.imshow(img)
plt.show()
