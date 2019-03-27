import numpy as np
import cv2
import time
import timeit


#cv2.useOptimized()
# Load two images
img1 = cv2.imread('/Users/juanzapata/opencv/samples/data/messi5.jpg')
e1 = cv2.getTickCount() #cv2.getTickCount function returns the number of clock-cycles after a reference event (like the moment machine was switched ON) to the moment this function is called.
for i in range(5,49,2):
    img1 = cv2.medianBlur(img1,i)
e2 = cv2.getTickCount() #cv2.getTickFrequency function returns the frequency of clock-cycles, or the number of clock-cycles per second.
t = (e2 - e1)/cv2.getTickFrequency()
print (t)
# # Result I got is 0.521107655 seconds

# Load two images
img1 = cv2.imread('/Users/juanzapata/opencv/samples/data/messi5.jpg')
e1 = time.time()
for i in range(5,49,2):
    img1 = cv2.medianBlur(img1,i)
e2 = time.time()
t = (e2 - e1)
print (t)
#
#
# # Python scalar operations are faster than Numpy scalar operations. So for operations including one or two elements, Python scalar is better than Numpy arrays. Numpy takes advantage when size of array is a little bit bigger.
#
# start_time = timeit.default_timer()
# x = 5; y = x**2
# print "OpenCV x**2",(timeit.default_timer() - start_time)
#
# start_time = timeit.default_timer()
# x = 5; y = x*x
# print "OpenCV x*x",(timeit.default_timer() - start_time)
#
# start_time = timeit.default_timer()
# x = np.uint8([5]); y = x*x
# print "Mix Numpy x*x",(timeit.default_timer() - start_time)
#
# start_time = timeit.default_timer()
# x = np.uint8([5]);y = np.square(x)
# print "Numpy square",(timeit.default_timer() - start_time)
#
# # Normally, OpenCV functions are faster than Numpy functions. So for same operation, OpenCV functions are preferred. But, there can be exceptions, especially when Numpy works with views instead of copies.
#
# img1 = cv2.imread('/Users/juanzapata/opencv/samples/data/messi5.jpg',0)
# start_time = timeit.default_timer()
# z = cv2.countNonZero(img1) #only for single channel array
# print "OpenCV",(timeit.default_timer() - start_time)
#
# start_time = timeit.default_timer()
# z = np.count_nonzero(img1)
# print "Numpy",(timeit.default_timer() - start_time)
#
#
# # cv2.imshow('res',img1)
# # #     #cv2.imshow() to display an image in a window. The window automatically fits to the image size.
# # cv2.waitKey(0)
# # #     #cv2.waitKey() is a keyboard binding function. Its argument is the time in milliseconds. The function waits for specified milliseconds for any keyboard event. If you press any key in that time, the program continues. If 0 is passed, it waits indefinitely for a key stroke. It can also be set to detect specific key strokes like, if key a is pressed etc which we will discuss below.
# # cv2.destroyAllWindows()
# # #     #cv2.destroyAllWindows() simply destroys all the windows we created. If you want to destroy any specific window, use the function cv2.destroyWindow() where you pass the exact window name as the argument.
