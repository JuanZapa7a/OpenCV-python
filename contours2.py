import cv2
import numpy as np

img = cv2.imread('images/star.jpg')
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
thresh, ret = cv2.threshold(img_gray, 127, 255, 0)
#contours,hierarchy = cv2.findContours(thresh,2,1)

contours, hierarchy = cv2.findContours(thresh,
                                       cv2.RETR_CCOMP,
                                       cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]


# Convexity Defects
# Remember we have to pass returnPoints = False while finding convex hull, in order to find convexity defects.

hull = cv2.convexHull(cnt,returnPoints = False)
defects = cv2.convexityDefects(cnt,hull)
# It returns an array where each row contains these values - [ start point, end point, farthest point, approximate distance to farthest point ]. We can visualize it using an image. We draw a line joining start point and end point, then draw a circle at the farthest point. Remember first three values returned are indices of cnt. So we have to bring those values from cnt.

for i in range(defects.shape[0]):
    s,e,f,d = defects[i,0]
    start = tuple(cnt[s][0])
    end = tuple(cnt[e][0])
    far = tuple(cnt[f][0])
    cv2.line(img,start,end,[0,255,0],2)
    cv2.circle(img,far,5,[0,0,255],-1)

# Point Polygon Test
# This function finds the shortest distance between a point in the image and a contour. It returns the distance which is negative when point is outside the contour, positive when point is inside and zero if point is on the contour. In the function cv2.pointPolygonTest, third argument is measureDist. If it is True, it finds the signed distance. If False, it finds whether the point is inside or outside or on the contour (it returns +1, -1, 0 respectively).

dist = cv2.pointPolygonTest(cnt,(50,50),True)
print(dist)

# Match Shapes
# OpenCV comes with a function cv2.matchShapes() which enables us to compare two shapes, or two contours and returns a metric showing the similarity. The lower the result, the better match it is. It is calculated based on the hu-moment values. Different measurement methods are explained in the docs.

img2 = cv2.imread('images/star2.jpg')
img2_gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
ret2, thresh2 = cv2.threshold(img2_gray, 127, 255,0)
contours,hierarchy = cv2.findContours(thresh,2,1)
cnt = contours[0]
contours,hierarchy = cv2.findContours(thresh2,2,1)
cnt2 = contours[0]

ret12 = cv2.matchShapes(cnt,cnt2,1,0.0)
print(ret12)
ret11 = cv2.matchShapes(cnt,cnt,1,0.0)
print(ret11)


cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
