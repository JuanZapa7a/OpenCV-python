import cv2
import numpy as np

img = cv2.imread('/Users/juanzapata/opencv/samples/data/messi5.jpg')


cv2.imshow('original',img)

# The size of the image can be specified manually, or you can specify the scaling factor. Different interpolation methods are used. Preferable interpolation methods are cv2.INTER_AREA for shrinking and cv2.INTER_CUBIC (slow) & cv2.INTER_LINEAR for zooming. By default, interpolation method used is cv2.INTER_LINEAR for all resizing purposes. You can resize an input image either of following methods:


res = cv2.resize(img,None,fx=2,fy=2, interpolation = cv2.INTER_CUBIC)
cv2.imshow('transf1',res)

#OR

height, width = img.shape[:2]
res2 = cv2.resize(img,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)

cv2.imshow('Transf2',res2)

img = cv2.imread('/Users/juanzapata/opencv/samples/data/messi5.jpg',0) #only one chanel
rows,cols = img.shape
M = np.float32([[1,0,100],[0,1,50]]) # Transmatrix M=([1 0 tx],[0 1 ty])

dst = cv2.warpAffine(img,M,(cols,rows))
cv2.imshow('Translated Image ',dst)


M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)# Transmatrix M=([alpha betha (1-alpha)cx-betha*cy],[-betha alpha betha*cx+(1-alpha)*cy])
dst = cv2.warpAffine(img,M,(cols,rows))
cv2.imshow('Rotated Image ',dst)

    #cv2.imshow() to display an image in a window. The window automatically fits to the image size.
cv2.waitKey(0)
    #cv2.waitKey() is a keyboard binding function. Its argument is the time in milliseconds. The function waits for specified milliseconds for any keyboard event. If you press any key in that time, the program continues. If 0 is passed, it waits indefinitely for a key stroke. It can also be set to detect specific key strokes like, if key a is pressed etc which we will discuss below.
cv2.destroyAllWindows()
    #cv2.destroyAllWindows() simply destroys all the windows we created. If you want to destroy any specific window, use the function cv2.destroyWindow() where you pass the exact window name as the argument.
