import numpy as np
import cv2
import os

# processing_dir = "SegmentationData/Domain1"
# processing_dir = "SegmentationData/Domain2"
# processing_dir = "SegmentationData/Domain3"
processing_dir = "SegmentationData/training"
os.makedirs(os.path.join(processing_dir, "mask"), exist_ok=True)
image_dir_path = os.path.join(processing_dir, "data")
image_list = os.listdir(image_dir_path)
print(image_list)

for im_name in image_list:
    image = cv2.imread(os.path.join(image_dir_path, im_name))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    interest = np.where(gray < 2)
    image = 255 * np.ones_like(image)
    image[interest] = np.array([0, 0, 0])
    size = image.shape[0:2]
    center = np.array(size) / 2
    pi = np.pi
    radius = np.min(size) // 2 * 0.95
    u = np.linspace(0, size[0]-1, size[0])
    v = np.linspace(0, size[1]-1, size[1])
    U, V = np.meshgrid(u, v)
    index = np.where((U-center[0])**2 + (V-center[1])**2 > radius**2)
    image[index[1], index[0]] = np.array([0, 0, 0])
    # for u in range(size[0]):
    #     for v in range(size[1]):
    #         if (u-center[0])**2 + (v-center[1])**2 > radius**2:
    #             image[u, v, :] = 0
    print(im_name)
    cv2.imwrite(os.path.join(processing_dir, 'mask',  os.path.basename(im_name).rsplit(".", maxsplit=1)[0] + ".png"), image)



# processing_dir = "Domain2"
# image_dir_path = os.path.join(processing_dir, "data")
# image_list = os.listdir(image_dir_path)
# print(image_list)
#
# for im_name in image_list:
#     image = cv2.imread(os.path.join(image_dir_path, im_name))
#     interest = np.where(image < 8)
#     image = 255 * np.ones_like(image)
#     image[interest] = 0
#     blue = np.where(image == 255)
#     # green = np.where(image == [0, 255, 0])
#     # red = np.where(image == [0, 0, 255])
#     yellow = np.where(image == [0, 255, 255])
#     image[yellow[0], yellow[1], yellow[2]] = 255
    # image[blue] = 0
    # image[green] = 0
    # image[red] = 0
    # cv2.imwrite(os.path.join(processing_dir, 'mask',  im_name), image)

# im = cv2.imread(os.path.join(image_dir_path, image_list[0]))
# im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray', im)
# cv2.waitKey(0)



