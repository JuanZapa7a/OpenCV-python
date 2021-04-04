import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# ffam = 'comic sans ns'
# fp = matplotlib.font_manager.FontProperties(
#     family=ffam, style='normal', size=10,
#     weight='normal', stretch='normal')

img = cv2.imread('images/dave.jpg')
# converting to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# remove noise
img = cv2.GaussianBlur(gray,(3,3),0)

laplacian = cv2.Laplacian(img,cv2.CV_64F)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
edges = cv2.Canny(img,100,200)

plt.subplot(2,3,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'),plt.xticks([]),plt.yticks([])
plt.subplot(2,3,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'),plt.xticks([]),plt.yticks([])
plt.subplot(2,3,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'),plt.xticks([]),plt.yticks([])
plt.subplot(2,3,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'),plt.xticks([]),plt.yticks([])
plt.subplot(2,3,5),plt.imshow(edges,cmap = 'gray')
plt.title('Canny'),plt.xticks([]),plt.yticks([])
plt.show()
