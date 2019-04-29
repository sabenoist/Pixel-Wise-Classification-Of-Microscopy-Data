import numpy as np
import scipy.ndimage as ndi
from skimage import io
from IPython.display import clear_output

import matplotlib.pyplot as plt

def make_weightmap(tup):

	(img_path, out_path, freqs, sigma) = tup

	clear_output(wait=True)
	print(img_path)

	img = io.imread(img_path)
	img = np.rollaxis(img, np.argmin(img.shape), 3)

	wmap = img.copy()
	wmap = wmap.astype(np.float32)
	wmap /= (freqs + 1)#[:,None]  # +1 to avoid zero division

	wmap[:,:,3] *= 2 # emphasise borders
	# wmap[:,:,2] *= 2 # borders is at index 2 now.

	wmap = .1 + np.sum(wmap, axis=np.argmin(img.shape)) * np.sum(ndi.filters.gaussian_filter(wmap, sigma), axis=np.argmin(img.shape))
	wmap[img[:,:,4]>0] = 0

	io.imsave(out_path, wmap)

	return wmap