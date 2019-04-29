from skimage import io

import skimage
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import numpy as np
import time
import math
import threading
import os



def convert_time_format(seconds):
	duration = list()
	duration.append(math.floor(seconds / 3600)) 
	duration.append(math.floor(seconds % 3600 / 60))
	duration.append(seconds % 3600 % 60)

	return duration


def preprocess_chunk(start, end, in_path, out_path):
	for i in range(start, end + 1):
		channels = read_image(in_path + "frame_%03d.tif" % (i))
		save_channels(channels, out_path, i)


def read_image(path):
	img = io.imread(path)
	channels = split_image_channels(img)
	
	return channels


def save_channels(channels, path, index):
	io.imsave(path + "cells/frame_%03d.tif" % (index), channels[0])
	io.imsave(path + "cell_borders/frame_%03d.tif" % (index), channels[1])
	io.imsave(path + "background/frame_%03d.tif" % (index), channels[2])


def plot_channels(channels, raw):
	print(raw.shape)

	fig = plt.figure()

	ax1 = fig.add_subplot(2,3,4) 
	ax1.set_title('cells')
	ax1.imshow(channels[0])
	
	ax2 = fig.add_subplot(2,3,5)
	ax2.set_title('cell_borders')
	ax2.imshow(channels[1])
	
	ax3 = fig.add_subplot(2,3,6)
	ax3.set_title('background')
	ax3.imshow(channels[2])
	

	ax4 = fig.add_subplot(2,3,2)
	ax4.imshow(raw, cmap = "bone")
	cm4 = plt.cm.get_cmap("rainbow")
	cm4.set_under('white')
	#ax4.imshow(channels[0])
	ax4.imshow(channels[1], cmap = cm4, alpha = 0.2, vmin=0.1, vmax=1)
	

	plt.show()


def image_to_binary(img):
	# print("image_to_binary() took %.2f seconds." % (time.time() - start))
	return 1 * (img > 0) 


def split_image_channels(img):
	img_cells = image_to_binary(img)
	img_borders = get_cell_borders(img_cells)
	img_background = get_img_background(img_cells, img_borders)

	return [img_cells, img_borders, img_background]


def get_img_background(cells, borders):
	# start = time.time()

	# background = cells.copy()
	#
	# for x in range(len(cells)):
	# 	for y in range(len(cells[0])):
	# 		if cells[x][y] == 1 or borders[x][y] == 1:
	# 			background[x][y] = 0
	# 		else:
	# 			background[x][y] = 1
	
	#print("get_img_background() took %.2f seconds." % (time.time() - start))
	# return 1 -(1 * (cells+borders) > 0)
	return ((cells+borders) == 0) *1
# /	return 11- * ~(cells or borders)


def get_cell_borders(img):
	# start = time.time()

	img_borders = img.copy()

	dilation_strength = 16
	for i in range(dilation_strength):
		img_borders = ndimage.binary_dilation(img_borders).astype(float)

	img_borders -= img

	# print("get_cell_borders() took %.2f seconds." % (time.time() - start))

	return img_borders


def show_menu():
	print("[1] read single image")
	print("[2] preprocess entire dataset")
	print("[x] exit")

	read_command()


def read_command():
	command = str(input("\noption: "))

	if command == "1":
		path = str(input("\npath: "))
		raw_path = str(input("\nraw_path: "))
		
		channels = read_image(path)
		raw = io.imread(raw_path)

		plot_channels(channels, raw)
	elif command == "2":
		preprocess_dataset()
	elif command == "x":
		global running
		running = False
	else:
		print("error: unknown command")
		read_command()


def preprocess_dataset():
	in_path = str(input("path in: "))
	out_path = str(input("path out: "))
	start = int(input("start number: "))
	end = int(input("end number: "))

	program_start = time.time()

	chunk_size = (end - start) // 8
	t1 = threading.Thread(target=preprocess_chunk, args=(start, start + chunk_size, in_path, out_path))
	t1.start()
	#print("chunk 1 starts at %d and ends at %d" % (start, start + chunk_size))

	start2 = start + chunk_size + 1
	end2 = start + 2 * chunk_size
	t2 = threading.Thread(target=preprocess_chunk, args=(start2, end2, in_path, out_path))
	t2.start()
	#print("chunk 2 starts at %d and ends at %d" % (start2, end2))

	start3 = end2 + 1
	end3 = start + 3 * chunk_size
	t3 = threading.Thread(target=preprocess_chunk, args=(start3, end3, in_path, out_path))
	t3.start()
	#print("chunk 3 starts at %d and ends at %d" % (start3, end3))

	start4 = end3 + 1
	end4 = start + 4 * chunk_size
	t4 = threading.Thread(target=preprocess_chunk, args=(start4, end4, in_path, out_path))
	t4.start()
	#print("chunk 4 starts at %d and ends at %d" % (start4, end4))

	start5 = end4 + 1
	end5 = start + 5 * chunk_size
	t5 = threading.Thread(target=preprocess_chunk, args=(start5, end5, in_path, out_path))
	t5.start()
	#print("chunk 5 starts at %d and ends at %d" % (start5, end5))

	start6 = end5 + 1
	end6 = start + 6 * chunk_size
	t6 = threading.Thread(target=preprocess_chunk, args=(start6, end6, in_path, out_path))
	t6.start()
	#print("chunk 6 starts at %d and ends at %d" % (start6, end6))

	start7 = end6 + 1
	end7 = start + 7 * chunk_size
	t7 = threading.Thread(target=preprocess_chunk, args=(start7, end7, in_path, out_path))
	t7.start()
	#print("chunk 7 starts at %d and ends at %d" % (start7, end7))

	start8 = end7 + 1
	end8 = end7 + (end - end7)
	t8 = threading.Thread(target=preprocess_chunk, args=(start8, end8, in_path, out_path))
	t8.start()
	#print("chunk 8 starts at %d and ends at %d" % (start8, end8))
	
	t1.join()
	t2.join()
	t3.join()
	t4.join()
	t5.join()
	t6.join()
	t7.join()
	t8.join()
	
	program_duration = convert_time_format(time.time() - program_start)
	print("Preprocessing the dataset took %d hours, %d minutes and %d seconds.\n" % (program_duration[0], program_duration[1], program_duration[2]))

	

running = True

while running:
	show_menu()
#path = "../datasets/20190326/segmented/frame_099.tif"

