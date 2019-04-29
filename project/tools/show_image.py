import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io

# img = io.imread("D:/Bachelor_Project/VU_Bachelor_Project/project/datasets/20190326/label/frame_000.tif")
img = io.imread("D:/Bachelor_Project/VU_Bachelor_Project/project/datasets/20190326/segmented_bordered/frame_000.tif")
# img = io.imread("D:/Bachelor_Project/VU_Bachelor_Project/project/datasets/20190326/patches/label/patch_000024.tif")

'''wmaps'''
# img = io.imread("D:/Bachelor_Project/VU_Bachelor_Project/project/datasets/20190326/patches/wmap/patch_007071.tif")
# img = io.imread("D:/Bachelor_Project/VU_Bachelor_Project/project/datasets/20190326/patches/wmap/patch_000024.tif")
# img = io.imread("D:/Bachelor_Project/VU_Bachelor_Project/project/datasets/20190326/patches/wmap/patch_006546.tif")
# img = io.imread("D:/Bachelor_Project/VU_Bachelor_Project/project/datasets/20190326/patches/wmap/patch_006274.tif")
# img = io.imread("D:/Bachelor_Project/VU_Bachelor_Project/project/datasets/20190326/patches/wmap/patch_003841.tif")



print(img.shape)

plt.imshow(img)
plt.show()


# for i in range(3):
#     plt.subplot(1, 3, i+1)
#     plt.imshow(img[:,:,i])
#
# plt.show()