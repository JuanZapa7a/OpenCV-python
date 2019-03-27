import cv2
import numpy as np


img = cv2.imread('blox.jpg')
rows,cols,ch = img.shape
# Making zoom
pts1 = np.float32([[51,65],[363,53],[28,387],[386,390]])
pts2 = np.float32([[0,0],[423,0],[0,419],[423,419]])

M = cv2.getPerspectiveTransform(pts1,pts2)
imgpersp = cv2.warpPerspective(img,M,(cols,rows))


gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
graypersp = cv2.cvtColor(imgpersp,cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
graypersp = np.float32(graypersp)
grayHarris = cv2.cornerHarris(gray,2,3,0.04)
grayperspHarris = cv2.cornerHarris(graypersp,2,3,0.04)
#result is dilated for marking the corners, not important
grayHarris = cv2.dilate(grayHarris,None)
grayperspHarris = cv2.dilate(grayperspHarris,None)

# Threshold for an optimal value, it may vary depending on the image.
img[grayHarris>0.01*grayHarris.max()]=[0,0,255]
imgpersp[grayperspHarris>0.01*grayperspHarris.max()]=[0,0,255]
cv2.imshow('dst',img)
cv2.imshow('dstpersp',imgpersp)

if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
