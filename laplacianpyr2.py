import cv2
import numpy as np
import pyplot as plt

img = cv2.imread('messi5.jpg')
img = cv2.resize(img,dsize = (512,512))

# generate Gaussian pyramid
G = img.copy()
gpA = [G]
for i in xrange(6):
    G = cv2.pyrDown(G)
    gpA.append(G)

# generate Laplacian Pyramid
lpA = [gpA[5]]
for i in xrange(5,0,-1):
    GE = cv2.pyrUp(gpA[i])
    L = cv2.subtract(gpA[i-1],GE)
    lpA.append(L)

cv2.imshow('Gaussian',lpA[5])
cv2.waitKey(0)
cv2.destroyAllWindows()
