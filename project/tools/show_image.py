import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io

img = io.imread('D:/Bachelor_Project/VU_Bachelor_Project/project/datasets/20190326/label/frame_000.tif')
# img = io.imread('D:/Bachelor_Project/VU_Bachelor_Project/project/datasets/20190326/segmented_bordered/frame_000.tif')
# img = io.imread('D:/Bachelor_Project/VU_Bachelor_Project/project/datasets/20190326/patches/label/patch_000024.tif')

'''wmaps'''
# img = io.imread('D:/Bachelor_Project/VU_Bachelor_Project/project/datasets/20190326/patches/wmap/patch_000084.tif')
# img = io.imread('D:/Bachelor_Project/VU_Bachelor_Project/project/datasets/20190326/patches/wmap/patch_000581.tif')
# img = io.imread('D:/Bachelor_Project/VU_Bachelor_Project/project/datasets/20190326/patches/wmap/patch_000660.tif')
# img = io.imread('D:/Bachelor_Project/VU_Bachelor_Project/project/datasets/20190326/patches/wmap/patch_000761.tif')
# img = io.imread('D:/Bachelor_Project/VU_Bachelor_Project/project/datasets/20190326/patches/wmap/patch_001016.tif')


# print(img.shape)
#
# plt.imshow(img)
# plt.show()


for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(img[:,:,i])

plt.show()