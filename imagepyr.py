import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('fruits.jpg')

print " Zoom In-Out demo "
print " Press u to zoom "
print " Press d to zoom "
print " Press esc to exit "

# img = cv2.imread('home.jpg')

while(1):
    h,w = img.shape[:2]

    cv2.imshow('image',img)
    k = cv2.waitKey(10)

    if k==27 :
        break

    elif k == ord('u'):  # Zoom in, make image double size
        img = cv2.pyrUp(img,dstsize = (2*w,2*h))

    elif k == ord('d'):  # Zoom down, make image half the size
        img = cv2.pyrDown(img,dstsize = (w/2,h/2))

cv2.destroyAllWindows()
