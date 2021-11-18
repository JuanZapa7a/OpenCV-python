import numpy as np
import matplotlib.pyplot as plt

import skimage.data as data
import skimage.segmentation as seg
import skimage.filters as filters
import skimage.draw as draw
import skimage.color as color
import skimage.io as io
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",
                default = "images/210908 10 H273.tif",
                help = "path to input image")
args = vars(ap.parse_args())

# Original
img = io.imread(args["image"])
# Cropping
rows,cols = img.shape
img = img[0:rows - 70, 0:cols]
titles = ['Image Original']
images = [img]

i = 0
fig, ax = plt.subplots(nrows=4,ncols=2)
plt.subplot(4, 2, 1+i)
plt.imshow(images[i],'gray')
plt.title(titles[i])
plt.xticks([]),plt.yticks([])


# Supervised Thresholding: Some prior knowledge, possibly
# from human input, is used to guide the algorithm.
text_segmented = img > 150
titles.append('Segmented T={}'.format(150))
images.append(text_segmented)

i = i+1
plt.subplot(4, 2, 1+i)
plt.imshow(images[i],'gray')
plt.title(titles[i])
plt.xticks([]),plt.yticks([])

# Unsupervised Thresholding: No prior knowledge is required.
# These algorithms attempt to subdivide images into
# meaningful regions automatically. The user may still be
# able to tweak certain settings to obtain desired outputs.

threshold = filters.threshold_otsu(img) # otsu,local,li,
# multi ...
text_threshold_otsu = img > threshold
titles.append('Unsupervised Otsu T={}'.format(threshold))
images.append(text_threshold_otsu)

i = i+1
plt.subplot(4, 2, 1+i)
plt.imshow(images[i],'gray')
plt.title(titles[i])
plt.xticks([]),plt.yticks([])

threshold = filters.threshold_local(img, 151, 'mean')
text_threshold_local = img > threshold
titles.append('Unsupervised Local')
images.append(text_threshold_local)

i = i+1
plt.subplot(4, 2, 1+i)
plt.imshow(images[i],'gray')
plt.title(titles[i])
plt.xticks([]),plt.yticks([])

img_gray = color.rgb2gray(img)
plt.imshow(img_gray, 'gray')

resolution = 360
radians = np.linspace(0, 2*np.pi, resolution)

radius = 300
X0 = 500
Y0 = 350

x = X0 + radius * np.cos(radians)  # cols
y = Y0 + radius * np.sin(radians)  # rows

circle_points = np.array([x, y]).T
points = circle_points[:-1]

titles.append('supervised Local')
images.append(img)

i = i+1
plt.subplot(4, 2, 1+i)
plt.plot(points[:, 0], points[:, 1], '--r', lw=3)
plt.axis('off')
plt.imshow(images[i],'gray')
plt.title(titles[i])
plt.xticks([]),plt.yticks([])


snake = seg.active_contour(img_gray, points)
titles.append('Default Snake')
images.append(img_gray)

i = i+1
plt.subplot(4, 2, 1+i)
plt.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
plt.axis('off')
plt.imshow(images[i],'gray')
plt.title(titles[i])
plt.xticks([]),plt.yticks([])

snake = seg.active_contour(img_gray, points,alpha=0.2,
                           beta=0.1)
titles.append('Custom Snake')
images.append(img_gray)

i = i+1
plt.subplot(4, 2, 1+i)
plt.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
plt.axis('off')
plt.imshow(images[i],'gray')
plt.title(titles[i])
plt.xticks([]),plt.yticks([])

plt.show()


