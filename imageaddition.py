import numpy as np
import cv2

# There is a difference between OpenCV addition and Numpy addition. OpenCV addition is a saturated operation while Numpy addition is a modulo operation.


# You can add two images by OpenCV function, cv2.add() or simply by numpy operation, res = img1 + img2. Both images should be of same depth and type, or second image can just be a scalar value.

x = np.uint8([250])
y = np.uint8([10])

print (cv2.add(x,y)) # 250+10 = 260 => [[255]]

print (x+y)          # 250+10 = 260 % 256 = [4]

img1 = cv2.imread('/Users/juanzapata/opencv/samples/data/ml.png')
img2 = cv2.imread('opencv_logo.png')
#print img1.[4]

dst = cv2.addWeighted(img1,0.7,img2,0.3,0)

cv2.imshow('dst',dst)
#     #cv2.imshow() to display an image in a window. The window automatically fits to the image size.
cv2.waitKey(0)
#     #cv2.waitKey() is a keyboard binding function. Its argument is the time in milliseconds. The function waits for specified milliseconds for any keyboard event. If you press any key in that time, the program continues. If 0 is passed, it waits indefinitely for a key stroke. It can also be set to detect specific key strokes like, if key a is pressed etc which we will discuss below.
cv2.destroyAllWindows()
#     #cv2.destroyAllWindows() simply destroys all the windows we created. If you want to destroy any specific window, use the function cv2.destroyWindow() where you pass the exact window name as the argument.
