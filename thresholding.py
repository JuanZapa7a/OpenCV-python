#%%
# Presentation on Rank filters.
import matplotlib.pyplot as plt
from skimage import data, io
from skimage.filters import try_all_threshold
import cv2
from PIL import Image
import numpy as np
import argparse
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default="images/210908 10 H273.tif",
	help="path to input image")
args = vars(ap.parse_args())

# img = data.page() #numpy array
img = io.imread(args["image"]) #numpy array
# cropping
rows, cols = img.shape
img = img[0:rows-70,0:cols]

fig, ax = try_all_threshold(img, figsize=(10, 8), verbose=False)
plt.show()

#%% THRESHOLD MEAN
# This example uses the mean value of pixel intensities. It is a simple and naive threshold value, which is sometimes used as a guess value.

from skimage.filters import threshold_mean

img = io.imread(args["image"]) #numpy array
# cropping
rows, cols = img.shape
img = img[0:rows-70,0:cols]

thresh = threshold_mean(img) # threshold value is the mean
binary = img > thresh

fig, axes = plt.subplots(ncols=2, figsize=(8, 3))
ax = axes.ravel()

ax[0].imshow(img, cmap=plt.cm.gray)
ax[0].set_title('Original image')

ax[1].imshow(binary, cmap=plt.cm.gray)
ax[1].set_title('Result')

for a in ax:
    a.axis('off')

plt.show()

#%% BIMODAL HISTOGRAM
from skimage.filters import threshold_minimum

#image = data.camera()
img = io.imread(args["image"]) #numpy array
# cropping
rows, cols = img.shape
img = img[0:rows-70,0:cols]

thresh_min = threshold_minimum(img)
binary_min = img > thresh_min

fig, ax = plt.subplots(2, 2, figsize=(10, 10))

ax[0, 0].imshow(img, cmap=plt.cm.gray)
ax[0, 0].set_title('Original')

ax[0, 1].hist(img.ravel(), bins=256)
ax[0, 1].set_title('Histogram')

ax[1, 0].imshow(binary_min, cmap=plt.cm.gray)
ax[1, 0].set_title('Thresholded (min)')

ax[1, 1].hist(img.ravel(), bins=256)
ax[1, 1].axvline(thresh_min, color='r')

for a in ax[:, 0]:
    a.axis('off')
plt.show()

#%% Otsu Threshold
from skimage.filters import threshold_otsu


#image = data.camera()
img = io.imread(args["image"]) #numpy array
# cropping
rows, cols = img.shape
img = img[0:rows-70,0:cols]

thresh = threshold_otsu(img)
binary = img > thresh

fig, axes = plt.subplots(ncols=3, figsize=(8, 2.5))
ax = axes.ravel()
ax[0] = plt.subplot(1, 3, 1, adjustable='box')
ax[1] = plt.subplot(1, 3, 2)
ax[2] = plt.subplot(1, 3, 3, sharex=ax[0], sharey=ax[0], adjustable='box')

ax[0].imshow(img, cmap=plt.cm.gray)
ax[0].set_title('Original')
ax[0].axis('off')

ax[1].hist(img.ravel(), bins=256)
ax[1].set_title('Histogram')
ax[1].axvline(thresh, color='r')

ax[2].imshow(binary, cmap=plt.cm.gray)
ax[2].set_title('Thresholded')
ax[2].axis('off')

plt.show()

#%% Local Threshold

from skimage.filters import threshold_otsu, threshold_local


#image = data.camera()
img = io.imread(args["image"]) #numpy array
# cropping
rows, cols = img.shape
img = img[0:rows-70,0:cols]

global_thresh = threshold_otsu(img)
binary_global = img > global_thresh

block_size = 35
adaptive_thresh = threshold_local(img, block_size, offset=10)
binary_adaptive = img > adaptive_thresh

fig, axes = plt.subplots(nrows=3, figsize=(7, 8))
ax = axes.ravel()
plt.gray()

ax[0].imshow(img)
ax[0].set_title('Original')

ax[1].imshow(binary_global)
ax[1].set_title('Global thresholding')

ax[2].imshow(binary_adaptive)
ax[2].set_title('Adaptive thresholding')

for a in ax:
    a.axis('off')

plt.show()

#%%  local threshold VS global threshold.

from skimage.morphology import disk
from skimage.filters import threshold_otsu, rank
from skimage.util import img_as_ubyte

# img = data.page() #numpy array
img = io.imread(args["image"]) #numpy array
# cropping
rows, cols = img.shape
img = img[0:rows-70,0:cols]
img = img_as_ubyte(img)

radius = 15
selem = disk(radius)

local_otsu = rank.otsu(img, selem)
threshold_global_otsu = threshold_otsu(img)
global_otsu = img >= threshold_global_otsu

fig, axes = plt.subplots(2, 2, figsize=(8, 5), sharex=True, sharey=True,
                         subplot_kw={'adjustable': 'box'})
ax = axes.ravel()
plt.tight_layout()

fig.colorbar(ax[0].imshow(img, cmap=plt.cm.gray), ax=ax[0],
             orientation='horizontal')
ax[0].set_title('Original')
ax[0].axis('off')

fig.colorbar(ax[1].imshow(local_otsu, cmap=plt.cm.gray), ax=ax[1],
             orientation='horizontal')
ax[1].set_title('Local Otsu (radius=%d)' % radius)
ax[1].axis('off')

ax[2].imshow(img >= local_otsu, cmap=plt.cm.gray)
ax[2].set_title('Original >= Local Otsu' % threshold_global_otsu)
ax[2].axis('off')

ax[3].imshow(global_otsu, cmap=plt.cm.gray)
ax[3].set_title('Global Otsu (threshold = %d)' % threshold_global_otsu)
ax[3].axis('off')

plt.show()