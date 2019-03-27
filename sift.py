import cv2

img = cv2.imread('home.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray, None)
cv2.imshow('Original', img)

img = cv2.drawKeypoints(img, kp, img)
# img=cv2.drawKeypoints(gray,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('SIFT', img)
k = cv2.waitKey(0) & 0xFF
# cv2.waitKey() is a keyboard binding function. Its argument is the time in milliseconds. The function waits for
# specified milliseconds for any keyboard event. If you press any key in that time, the program continues. If 0 is
# passed, it waits indefinitely for a key stroke. It can also be set to detect specific key strokes like, if key a is
# pressed etc which we will discuss below.
if k == 27:
    cv2.destroyAllWindows()
# cv2.destroyAllWindows() simply destroys all the windows we created. If you want to destroy any specific window,
# use the function cv2.destroyWindow() where you pass the exact window name as the argument.




# import cv2
# import numpy as np
#
# img = cv2.imread('home.jpg')
# gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#
# sift = cv2.xfeatures2d.SIFT_create()
# kp = sift.detect(gray,None)
#
# imgsift=cv2.drawKeypoints(gray,kp,imgsift)
#
# # cv2.imwrite('sift_keypoints.jpg',img)
# cv2.imshow('Org',img)
# cv2.imshow('sift',imgsift)
#
# if cv2.waitKey(0) & 0xff == 27:
#     cv2.destroyAllWindows()