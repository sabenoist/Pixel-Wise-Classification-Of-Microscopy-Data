from pims import ND2_Reader as nd2
from skimage import io
import numpy as np
import os
from IPython.display import clear_output

def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def stack_to_image_list(stack_path, out_path, z=0, channel=0, position=0):
	stack = nd2(stack_path)
	for f in range(stack.sizes['t']):
		clear_output(wait=True)
		print(f)
		img = stack.get_frame_2D(c=channel,t=f,m=position,z=z)
		io.imsave('{}/frame_{}.tif'.format(out_path, str(f).zfill(3)), img.astype(np.float32))

def classification_to_one_hot_ground_truth(prediction_path, out_path, number_of_classes=None):
	files = [a for a in os.listdir(prediction_path) if a.endswith('.tif')]

	for f in files:
		img = io.imread('{}/{}'.format(prediction_path, f))
		img_argmax = np.argmax(img, axis=0)

		if number_of_classes is None:
			noc = np.max(img_argmax) + 1
		else:
			noc = number_of_classes

		img_out = np.eye(noc)[img_argmax]
		io.imsave('{}/{}'.format(out_path, f), img_out.astype(np.uint8))

	return