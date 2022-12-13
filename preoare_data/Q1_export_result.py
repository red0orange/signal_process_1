import os
import shutil
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from red0orange.file import *
from utils import image_domain_transfer


if __name__ == "__main__":
    result_save_root = "/media/red0orange/Data/Ubuntu-Insync/Courses/现代信号处理/Proj1/Results/Q1"
    os.makedirs(os.path.join(result_save_root, "0"), exist_ok=True)
    os.makedirs(os.path.join(result_save_root, "1"), exist_ok=True)
    os.makedirs(os.path.join(result_save_root, "2"), exist_ok=True)

    root = "/media/red0orange/Data/Ubuntu-Insync/Courses/现代信号处理/Proj1/Data"
    source_image_path = os.path.join(root, "source.jpg")
    source_image = cv2.imread(source_image_path)
    domain_0_root = os.path.join(root, "0")
    domain_1_root = os.path.join(root, "1")
    domain_2_root = os.path.join(root, "2")
    domain_0_data = get_image_files(domain_0_root)
    domain_1_data = get_image_files(domain_1_root)
    domain_2_data = get_image_files(domain_2_root)


    # obtain and record domain_transfer_image
    L = 0.003
    lam = 1 - 0.6
    resize_shape = [384, 384]
    transfer_images_dict = {0: [], 1: [], 2: []}
    for i, save_root, target_domain_data in zip([0, 1, 2], [os.path.join(result_save_root, "0"), os.path.join(result_save_root, "1"), os.path.join(result_save_root, "2")], [domain_0_data, domain_1_data, domain_2_data]):
        for target_image_path in target_domain_data:
            transfer_image_save_path = os.path.join(save_root, os.path.basename(target_image_path).rsplit(".", maxsplit=1)[0] + ".png")
            transfer_image_f_save_path = os.path.join(save_root, os.path.basename(target_image_path).rsplit(".", maxsplit=1)[0] + "_f.png")
            target_image = cv2.imread(target_image_path)
            transfer_image, transfer_image_f = image_domain_transfer(source_image, target_image, L, lam, reshape_shape=resize_shape, return_f=True)
            cv2.imwrite(transfer_image_save_path, transfer_image)
            cv2.imwrite(transfer_image_f_save_path, transfer_image_f)
            transfer_images_dict[i].append(transfer_image)

    # measure inter-class distance
    domain_01_psnr_distance = np.zeros(shape=[len(transfer_images_dict[0]), len(transfer_images_dict[1])])
    domain_02_psnr_distance = np.zeros(shape=[len(transfer_images_dict[0]), len(transfer_images_dict[2])])
    domain_12_psnr_distance = np.zeros(shape=[len(transfer_images_dict[1]), len(transfer_images_dict[2])])
    domain_01_ssim_distance = np.zeros(shape=[len(transfer_images_dict[0]), len(transfer_images_dict[1])])
    domain_02_ssim_distance = np.zeros(shape=[len(transfer_images_dict[0]), len(transfer_images_dict[2])])
    domain_12_ssim_distance = np.zeros(shape=[len(transfer_images_dict[1]), len(transfer_images_dict[2])])
    for result_arrs, images in zip([[domain_01_psnr_distance, domain_01_ssim_distance], [domain_02_psnr_distance, domain_02_ssim_distance], [domain_12_psnr_distance, domain_12_ssim_distance]], [[transfer_images_dict[0], transfer_images_dict[1]], [transfer_images_dict[0], transfer_images_dict[2]], [transfer_images_dict[1], transfer_images_dict[2]]]):
        for i, image_i in enumerate(images[0]):
            image_i = image_i.astype(np.uint8)
            for j, image_j in enumerate(images[1]):
                image_j = image_j.astype(np.uint8)
                ssim_distance = ssim(image_i, image_j, channel_axis=2)
                result_arrs[1][i, j] = ssim_distance
                psnr_distance = psnr(image_i, image_j)
                result_arrs[0][i, j] = psnr_distance

    print("Mean PSNR between domain 0 and domain 1: ", np.mean(domain_01_psnr_distance))
    print("Min PSNR between domain 0 and domain 1: ", np.min(domain_01_psnr_distance))
    print("Max PSNR between domain 0 and domain 1: ", np.max(domain_01_psnr_distance))
    print("Mean PSNR between domain 0 and domain 2: ", np.mean(domain_02_psnr_distance))
    print("Min PSNR between domain 0 and domain 2: ", np.min(domain_02_psnr_distance))
    print("Max PSNR between domain 0 and domain 2: ", np.max(domain_02_psnr_distance))
    print("Mean PSNR between domain 1 and domain 2: ", np.mean(domain_12_psnr_distance))
    print("Min PSNR between domain 1 and domain 2: ", np.min(domain_12_psnr_distance))
    print("Max PSNR between domain 1 and domain 2: ", np.max(domain_12_psnr_distance))

    print("Mean SSIM between domain 0 and domain 1: ", np.mean(domain_01_ssim_distance))
    print("Min SSIM between domain 0 and domain 1: ", np.min(domain_01_ssim_distance))
    print("Max SSIM between domain 0 and domain 1: ", np.max(domain_01_ssim_distance))
    print("Mean SSIM between domain 0 and domain 2: ", np.mean(domain_02_ssim_distance))
    print("Min SSIM between domain 0 and domain 2: ", np.min(domain_02_ssim_distance))
    print("Max SSIM between domain 0 and domain 2: ", np.max(domain_02_ssim_distance))
    print("Mean SSIM between domain 1 and domain 2: ", np.mean(domain_12_ssim_distance))
    print("Min SSIM between domain 1 and domain 2: ", np.min(domain_12_ssim_distance))
    print("Max SSIM between domain 1 and domain 2: ", np.max(domain_12_ssim_distance))
    pass