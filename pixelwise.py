import cv2
import numpy as np

img = cv2.imread('/home/juanzapata/OpenCV/samples/data/messi5.jpg',1)

px = img[100,100]
print px

# accessing only blue pixel
pxblue = img[100,100,0]
print pxblue

img[100,100] = [255,255,255]
print img[100,100]

    #Warning Numpy is a optimized library for fast array calculations. So simply accessing each and every pixel values and modifying it will be very slow and it is discouraged. For individual pixel access, Numpy array methods, array.item() and array.itemset() is considered to be better. But it always returns a scalar. So if you want to access all B,G,R values, you need to call array.item() separately for all.

# accessing only red pixel
print img.item(100,100,2)
img.itemset((100,100,2),100)
print img.item(100,100,2)

# image properties
print img.shape
print img.size
print img.dtype


#image ROI
ball = img[280:340, 330:390]
img[273:333, 100:160] = ball

#Splitting and Merging Image Channels
#b,g,r = cv2.split(img)
#img = cv2.merge((b,g,r))

b = img[:,:,0]
g = img[:,:,1]
r = img[:,:,2]

b[:,:]=0
cv2.imshow('blue',b)
cv2.imshow('green',g)
cv2.imshow('red',r)
cv2.imshow('pixelwise',img)

cv2.waitKey(0)

cv2.destroyAllWindows()
