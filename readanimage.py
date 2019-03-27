import cv2 # Load openCV
import numpy as np # Load Numpy a Base N-dimensional array package
# from matplotlib import pyplot as plt


#load an color image in grayscale
# img = cv2.imread('/home/juanzapata/OpenCV/samples/data/messi5.jpg',1)
img = cv2.imread('/Users/juanzapata/MEGA/Downloads/opencv/samples/data/messi5.jpg',1)
    #cv2.IMREAD_COLOR : Loads a color image. Any transparency of image will be neglected. It is the default flag. -1
    #cv2.IMREAD_GRAYSCALE : Loads image in grayscale mode. 0
    #cv2.IMREAD_UNCHANGED : Loads image as such including alpha channel. 1

cv2.imshow('image',img)
    #cv2.imshow() to display an image in a window. The window automatically fits to the image size.
cv2.waitKey(0)
    #cv2.waitKey() is a keyboard binding function. Its argument is the time in milliseconds. The function waits for specified milliseconds for any keyboard event. If you press any key in that time, the program continues. If 0 is passed, it waits indefinitely for a key stroke. It can also be set to detect specific key strokes like, if key a is pressed etc which we will discuss below.
cv2.destroyAllWindows()
    #cv2.destroyAllWindows() simply destroys all the windows we created. If you want to destroy any specific window, use the function cv2.destroyWindow() where you pass the exact window name as the argument.
