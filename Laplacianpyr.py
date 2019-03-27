import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('messi5.jpg')
img = cv2.resize(img,dsize = (512,512))
imgl1d = cv2.pyrDown(img)
imgl2d = cv2.pyrDown(imgl1d)
imgl3d = cv2.pyrDown(imgl2d)
imgl2u = cv2.pyrUp(imgl3d)
imgl1u = cv2.pyrUp(imgl2u)
imgl0 = cv2.pyrUp(imgl1u)


#grayl0 = cv2.cvtColor(img-imgl0d,cv2.COLOR_BGR2GRAY)

# gray_lap = cv2.Laplacian(grayl0,ddepth,ksize = kernel_size,scale = scale,delta = delta)
# dst = cv2.convertScaleAbs(gray_lap)
# dst = cv2.convertScaleAbs(cv2.absdiff(img,imgl0))
# cv2.imshow('laplacian L0',cv2.absdiff(img,imgl0d))
# cv2.imshow('laplacian L0',cv2.absdiff(imgl1u,imgl1d))
# cv2.imshow('laplacian',dst)

print img.shape[:2]
print imgl1d.shape[:2]
print imgl2d.shape[:2]
print imgl3d.shape[:2]
print imgl2u.shape[:2]
print imgl1u.shape[:2]
print imgl0.shape[:2]
cv2.imshow('laplacian 0',cv2.cvtColor(cv2.absdiff(img,imgl0),cv2.COLOR_BGR2GRAY))
# cv2.imshow('laplacian 1',cv2.absdiff(imgl1u,imgl1d))
cv2.waitKey(0)
cv2.destroyAllWindows()
