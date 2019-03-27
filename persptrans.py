import cv2
import numpy as np
from matplotlib import pyplot as plt

# img = cv2.imread('/Users/juanzapata/opencv/samples/data/sudoku.png',0)
img = cv2.imread('sudoku.jpg')
rows,cols,ch = img.shape

# AFFINE
# In affine transformation, all parallel lines in the original image will still be parallel in the output image. To find the transformation matrix, we need three points from input image and their corresponding locations in output image. Then cv2.getAffineTransform will create a 2x3 matrix which is to be passed to cv2.warpAffine.
pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])

M = cv2.getAffineTransform(pts1,pts2)
dst = cv2.warpAffine(img,M,(cols,rows))

# PERSPETIVE
# For perspective transformation, you need a 3x3 transformation matrix. Straight lines will remain straight even after the transformation. To find this transformation matrix, you need 4 points on the input image and corresponding points on the output image. Among these 4 points, 3 of them should not be collinear. Then transformation matrix can be found by the function cv2.getPerspectiveTransform. Then apply cv2.warpPerspective with this 3x3 transformation matrix.
pts1 = np.float32([[51,65],[363,53],[28,387],[386,390]])
pts2 = np.float32([[0,0],[423,0],[0,419],[423,419]])

M = cv2.getPerspectiveTransform(pts1,pts2)
dst2 = cv2.warpPerspective(img,M,(cols,rows))

cv2.imshow('Input',img)
cv2.imshow('Affine Output',dst)
cv2.imshow('Persoective Output',dst2)


# plt.subplot(131),plt.imshow(img),plt.title('Input')
# plt.subplot(132),plt.imshow(dst),plt.title('Affine Output')
# plt.subplot(133),plt.imshow(dst2),plt.title('Perspective Output')
# plt.show()

if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
