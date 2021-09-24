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
ap.add_argument("-i", "--image", default="images/210908 10 H262.tif",
	help="path to input image")
args = vars(ap.parse_args())

# img = data.page() #numpy array
img = io.imread(args["image"])
#shape = img.shape
#img = img[0:rows-70,0:cols]
fig, ax = try_all_threshold(img, figsize=(10, 8), verbose=False)
plt.show()

#%%
