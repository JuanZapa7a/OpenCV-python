import numpy as np
import cv2
import random

im = cv2.imread('images/messi5.jpg')
imblur = cv2.GaussianBlur(im,(5,5),0)
imgray = cv2.cvtColor(imblur,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
# Contours can be explained simply as a curve joining all the continuous points (along the boundary), having same color or intensity. The contours are a useful tool for shape analysis and object detection and recognition.

        # For better accuracy, use binary images. So before finding contours, apply threshold or canny edge detection.
        # findContours function modifies the source image. So if you want source image even after finding contours, already store it to some other variables.
        # In OpenCV, finding contours is like finding white object from black background. So remember, object to be found should be white and background should be black.

_,contours,hierarchy= cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# RETR_LIST
# It simply retrieves all the contours, but doesn’t create any parent-child relationship. Parents and kids are equal under this rule, and they are just contours. ie they all belongs to same hierarchy level.

# RETR_EXTERNAL
# If you use this flag, it returns only extreme outer flags. All child contours are left behind. We can say, under this law, Only the eldest in every family is taken care of. It doesn’t care about other members of the family :).

# RETR_CCOMP
# This flag retrieves all the contours and arranges them to a 2-level hierarchy. ie external contours of the object (ie its boundary) are placed in hierarchy-1. And the contours of holes inside object (if any) is placed in hierarchy-2. If any object inside it, its contour is placed again in hierarchy-1 only. And its hole in hierarchy-2 and so on.

# RETR_TREE
# And this is the final guy, Mr.Perfect. It retrieves all the contours and creates a full family hierarchy list. It even tells, who is the grandpa, father, son, grandson and even beyond... :).

cnt = contours[100]
M = cv2.moments(cnt)
print (M)
print (len(contours))

############
# FEATURES #
############

# Centroid
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])
# Area
area = cv2.contourArea(cnt)
# Perimeter
perimeter = cv2.arcLength(cnt,True)
# Approx
epsilon = 0.1*cv2.arcLength(cnt,True)
approx = cv2.approxPolyDP(cnt,epsilon,True)
# Convex Hull
hull = cv2.convexHull(cnt)
# Convexity
k = cv2.isContourConvex(cnt)
# Bounding Rectangle
x,y,w,h = cv2.boundingRect(cnt)
cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
# Rotated boundingRect
rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)
cv2.drawContours(im,[box],0,(0,0,255),2)
# Minimum Enclosing Circle
(x,y),radius = cv2.minEnclosingCircle(cnt)
center = (int(x),int(y))
radius = int(radius)
cv2.circle(im,center,radius,(0,255,0),2)
# Fitting an Ellipse
ellipse = cv2.fitEllipse(cnt)
cv2.ellipse(im,ellipse,(0,255,0),2)
# Fitting a Line
rows,cols = im.shape[:2]
[vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
lefty = int((-x*vy/vx) + y)
righty = int(((cols-x)*vy/vx)+y)
cv2.line(im,(cols-1,righty),(0,lefty),(0,255,0),2)

##############
# PROPERTIES #
##############

# Aspect Ratio
aspect_ratio = float(w)/h

# Extent
rect_area = w*h
extent = float(area)/rect_area

# Solidity
hull_area = cv2.contourArea(hull)
solidity = float(area)/hull_area

# Equivalent Diameter
equi_diameter = np.sqrt(4*area/np.pi)

# Orientation
(x,y),(MA,ma),angle = cv2.fitEllipse(cnt)

# Mask and Pixel Points
mask = np.zeros(imgray.shape,np.uint8)
cv2.drawContours(mask,[cnt],0,255,-1)
pixelpoints = np.transpose(np.nonzero(mask))
#pixelpoints = cv2.findNonZero(mask)

# Maximum Value, Minimum Value and their locations
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(imgray,mask = mask)

# Mean Color or Mean Intensity
mean_val = cv2.mean(im,mask = mask)

# Extreme Points
leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])

for i in range(0, len(hierarchy[0])):
    # cv2.scalar_color(rand()&255, rand()&255, rand()&255)
    cv2.drawContours(im, contours, i, (random.random()*255, random.random()*255, random.random()*255), cv2.FILLED, 1, hierarchy)

cv2.imshow('image',im)
    #cv2.imshow() to display an image in a window. The window automatically fits to the image size.
cv2.waitKey(0)
    #cv2.waitKey() is a keyboard binding function. Its argument is the time in milliseconds. The function waits for specified milliseconds for any keyboard event. If you press any key in that time, the program continues. If 0 is passed, it waits indefinitely for a key stroke. It can also be set to detect specific key strokes like, if key a is pressed etc which we will discuss below.
cv2.destroyAllWindows()
    #cv2.destroyAllWindows() simply destroys all the windows we created. If you want to destroy any specific window, use the function cv2.destroyWindow() where you pass the exact window name as the argument.
