import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io

# img = io.imread('D:/Bachelor_Project/VU_Bachelor_Project/project/datasets/20190326/label/frame_000.tif')
# img = io.imread('D:/Bachelor_Project/VU_Bachelor_Project/project/datasets/20190326/segmented_bordered/frame_000.tif')
# img = io.imread('D:/Bachelor_Project/VU_Bachelor_Project/project/datasets/20190326/patches/label/patch_000024.tif')

'''wmaps'''
# img = io.imread('D:/Bachelor_Project/VU_Bachelor_Project/project/datasets/20190326/patches/wmap/patch_000060.tif')
# img = io.imread('D:/Bachelor_Project/VU_Bachelor_Project/project/datasets/20190326/patches/wmap/patch_000108.tif')
# img = io.imread('D:/Bachelor_Project/VU_Bachelor_Project/project/datasets/20190326/patches/wmap/patch_000158.tif')
# img = io.imread('D:/Bachelor_Project/VU_Bachelor_Project/project/datasets/20190326/patches/wmap/patch_000227.tif')
img = io.imread('D:/Bachelor_Project/VU_Bachelor_Project/project/datasets/20190326/patches/wmap/patch_000266.tif')
img_raw = io.imread('D:/Bachelor_Project/VU_Bachelor_Project/project/datasets/20190326/patches/raw/patch_000266.tif')


print(img.shape)

plt.subplot(1, 2, 1)
plt.imshow(img)

plt.subplot(1, 2, 2)
plt.imshow(img_raw)
plt.show()


# for i in range(5):
#     plt.subplot(1, 5, i+1)
#     plt.imshow(img[:,:,i])
#
# plt.show()