import os
import csv
import multiprocessing

import numpy as np
import cv2

from red0orange.file import *
from utils import image_domain_transfer


enhance_save_root = "SegmentationData/Enhance_training"


def process(data, domain_1_data, domain_2_data, domain_3_data, lams, L, enhance_save_root, resize_shape):
    index = 0
    image_path = data[0]
    label_path = data[1]
    mask_path = data[2]

    for domain in [domain_1_data, domain_2_data, domain_3_data]:
        for j, (target_image_path, _) in enumerate(domain):
            for lam in lams:
                print("{} Current Index: ".format(os.path.basename(image_path)), index)
                image = cv2.imread(image_path)
                target_image = cv2.imread(target_image_path)
                label_image = cv2.imread(label_path)
                mask_image = cv2.imread(mask_path)
                transfer_image = image_domain_transfer(image, target_image, L, lam, reshape_shape=resize_shape)
                
                transfer_image_save_path = os.path.join(enhance_save_root, "data", "{}_{}.png".format(os.path.basename(image_path).rsplit(".", maxsplit=1)[0], index))
                label_image_save_path = os.path.join(enhance_save_root, "label", "{}_{}.png".format(os.path.basename(image_path).rsplit(".", maxsplit=1)[0], index))
                mask_image_save_path = os.path.join(enhance_save_root, "mask", "{}_{}.png".format(os.path.basename(image_path).rsplit(".", maxsplit=1)[0], index))

                cv2.imwrite(transfer_image_save_path, transfer_image)
                cv2.imwrite(label_image_save_path, cv2.resize(label_image, resize_shape))
                cv2.imwrite(mask_image_save_path, cv2.resize(mask_image, resize_shape))

                index += 1
            pass


if __name__ == "__main__":
    num_process = 80
    enhance_save_root = "SegmentationData/Enhance_training"
    os.makedirs(enhance_save_root, exist_ok=True)
    os.makedirs(os.path.join(enhance_save_root, "data"), exist_ok=True)
    os.makedirs(os.path.join(enhance_save_root, "label"), exist_ok=True)
    os.makedirs(os.path.join(enhance_save_root, "mask"), exist_ok=True)

    root = "SegmentationData"
    training_root = os.path.join(root, "training")
    domain_1_root = os.path.join(root, "Domain1")
    domain_2_root = os.path.join(root, "Domain2")
    domain_3_root = os.path.join(root, "Domain3")

    def sort_func(i):
        return i.rsplit(".", maxsplit=1)[0]

    training_data = list(zip(sorted(get_image_files(os.path.join(training_root, "data")), key=sort_func), sorted(get_image_files(os.path.join(training_root, "label")), key=sort_func), sorted(get_image_files(os.path.join(training_root, "mask")), key=sort_func)))
    domain_1_data = list(zip(sorted(get_image_files(os.path.join(domain_1_root, "data"))), sorted(get_image_files(os.path.join(domain_1_root, "label")))))
    domain_2_data = list(zip(sorted(get_image_files(os.path.join(domain_2_root, "data"))), sorted(get_image_files(os.path.join(domain_2_root, "label")))))
    domain_3_data = list(zip(sorted(get_image_files(os.path.join(domain_3_root, "data"))), sorted(get_image_files(os.path.join(domain_3_root, "label")))))

    Select domain 3 for enhance
    L = 0.003
    lams = [0.3, 0.6, 0.9]
    resize_shape = [384, 384]

    pool = multiprocessing.Pool(num_process)
    processes = []
    for i, (image_path, label_path, mask_path) in enumerate(training_data):
        processes.append(pool.apply_async(func=process, args=([image_path, label_path, mask_path], domain_1_data, domain_2_data, domain_3_data, lams, L, enhance_save_root, resize_shape)))
    pool.close()
    pool.join()

    for i, (image_path, label_path, mask_path) in enumerate(training_data):
        image_save_path = os.path.join(enhance_save_root, "data", "{}.png".format(os.path.basename(image_path).rsplit(".", maxsplit=1)[0]))
        label_image_save_path = os.path.join(enhance_save_root, "label", "{}.png".format(os.path.basename(image_path).rsplit(".", maxsplit=1)[0]))
        mask_image_save_path = os.path.join(enhance_save_root, "mask", "{}.png".format(os.path.basename(image_path).rsplit(".", maxsplit=1)[0]))

        image = cv2.imread(image_path)
        label_image = cv2.imread(label_path)
        mask_image = cv2.imread(mask_path)

        cv2.imwrite(image_save_path, cv2.resize(image, resize_shape))
        cv2.imwrite(label_image_save_path, cv2.resize(label_image, resize_shape))
        cv2.imwrite(mask_image_save_path, cv2.resize(mask_image, resize_shape))