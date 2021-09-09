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
