import os
import shutil
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from red0orange.file import *
from prepare_data.utils import image_domain_transfer


if __name__ == "__main__":
    L = 0.003
    lam = 1 - 0.9
    resize_shape = [384, 384]

    result_save_root = "/home/dehao/github_projects/pro/Results/Q1-0.9"
    os.makedirs(os.path.join(result_save_root, "0"), exist_ok=True)
    os.makedirs(os.path.join(result_save_root, "1"), exist_ok=True)
    os.makedirs(os.path.join(result_save_root, "2"), exist_ok=True)

    root = "/home/dehao/github_projects/pro/Data"
    source_image_path = os.path.join(root, "source.jpg")
    source_image = cv2.imread(source_image_path)
    source_image = cv2.resize(source_image, resize_shape)
    domain_0_root = os.path.join(root, "0")
    domain_1_root = os.path.join(root, "1")
    domain_2_root = os.path.join(root, "2")
    domain_0_data = get_image_files(domain_0_root)
    domain_1_data = get_image_files(domain_1_root)
    domain_2_data = get_image_files(domain_2_root)
    ori_image_dict = {0: [], 1: [], 2: []}
    for i, domain_image_paths in enumerate([domain_0_data, domain_1_data, domain_2_data]):
        for image_path in domain_image_paths:
            ori_image_dict[i].append(cv2.resize(cv2.imread(image_path), resize_shape))
        pass

    # obtain and record domain_transfer_image
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
    source_domain_0_psnr_distance = np.zeros(shape=[1, len(transfer_images_dict[0])])
    source_domain_0_ssim_distance = np.zeros(shape=[1, len(transfer_images_dict[0])])
    source_domain_1_psnr_distance = np.zeros(shape=[1, len(transfer_images_dict[1])])
    source_domain_1_ssim_distance = np.zeros(shape=[1, len(transfer_images_dict[1])])
    source_domain_2_psnr_distance = np.zeros(shape=[1, len(transfer_images_dict[2])])
    source_domain_2_ssim_distance = np.zeros(shape=[1, len(transfer_images_dict[2])])
    domain_0_transfer_psnr_distance = np.zeros(shape=[len(transfer_images_dict[0]), len(transfer_images_dict[0])])
    domain_0_transfer_ssim_distance = np.zeros(shape=[len(transfer_images_dict[0]), len(transfer_images_dict[0])])
    domain_1_transfer_psnr_distance = np.zeros(shape=[len(transfer_images_dict[1]), len(transfer_images_dict[1])])
    domain_1_transfer_ssim_distance = np.zeros(shape=[len(transfer_images_dict[1]), len(transfer_images_dict[1])])
    domain_2_transfer_psnr_distance = np.zeros(shape=[len(transfer_images_dict[2]), len(transfer_images_dict[2])])
    domain_2_transfer_ssim_distance = np.zeros(shape=[len(transfer_images_dict[2]), len(transfer_images_dict[2])])
    domain_0_self_psnr_distance = np.zeros(shape=[len(transfer_images_dict[0]), len(transfer_images_dict[0])])
    domain_0_self_ssim_distance = np.zeros(shape=[len(transfer_images_dict[0]), len(transfer_images_dict[0])])
    domain_1_self_psnr_distance = np.zeros(shape=[len(transfer_images_dict[1]), len(transfer_images_dict[1])])
    domain_1_self_ssim_distance = np.zeros(shape=[len(transfer_images_dict[1]), len(transfer_images_dict[1])])
    domain_2_self_psnr_distance = np.zeros(shape=[len(transfer_images_dict[2]), len(transfer_images_dict[2])])
    domain_2_self_ssim_distance = np.zeros(shape=[len(transfer_images_dict[2]), len(transfer_images_dict[2])])


    domain_01_psnr_distance = np.zeros(shape=[len(transfer_images_dict[0]), len(transfer_images_dict[1])])
    domain_02_psnr_distance = np.zeros(shape=[len(transfer_images_dict[0]), len(transfer_images_dict[2])])
    domain_12_psnr_distance = np.zeros(shape=[len(transfer_images_dict[1]), len(transfer_images_dict[2])])
    domain_01_ssim_distance = np.zeros(shape=[len(transfer_images_dict[0]), len(transfer_images_dict[1])])
    domain_02_ssim_distance = np.zeros(shape=[len(transfer_images_dict[0]), len(transfer_images_dict[2])])
    domain_12_ssim_distance = np.zeros(shape=[len(transfer_images_dict[1]), len(transfer_images_dict[2])])
    ori_domain_01_psnr_distance = np.zeros(shape=[len(transfer_images_dict[0]), len(transfer_images_dict[1])])
    ori_domain_02_psnr_distance = np.zeros(shape=[len(transfer_images_dict[0]), len(transfer_images_dict[2])])
    ori_domain_12_psnr_distance = np.zeros(shape=[len(transfer_images_dict[1]), len(transfer_images_dict[2])])
    ori_domain_01_ssim_distance = np.zeros(shape=[len(transfer_images_dict[0]), len(transfer_images_dict[1])])
    ori_domain_02_ssim_distance = np.zeros(shape=[len(transfer_images_dict[0]), len(transfer_images_dict[2])])
    ori_domain_12_ssim_distance = np.zeros(shape=[len(transfer_images_dict[1]), len(transfer_images_dict[2])])
    for result_arrs, images in zip(
        [
            [source_domain_0_psnr_distance, source_domain_0_ssim_distance],
            [source_domain_1_psnr_distance, source_domain_1_ssim_distance],
            [source_domain_2_psnr_distance, source_domain_2_ssim_distance],
            [domain_0_transfer_psnr_distance, domain_0_transfer_ssim_distance],
            [domain_1_transfer_psnr_distance, domain_1_transfer_ssim_distance],
            [domain_2_transfer_psnr_distance, domain_2_transfer_ssim_distance],
            [domain_0_self_psnr_distance, domain_0_self_ssim_distance],
            [domain_1_self_psnr_distance, domain_1_self_ssim_distance],
            [domain_2_self_psnr_distance, domain_2_self_ssim_distance],
            [domain_01_psnr_distance, domain_01_ssim_distance],
            [domain_02_psnr_distance, domain_02_ssim_distance],
            [domain_12_psnr_distance, domain_12_ssim_distance],
            [ori_domain_01_psnr_distance, ori_domain_01_ssim_distance],
            [ori_domain_02_psnr_distance, ori_domain_02_ssim_distance],
            [ori_domain_12_psnr_distance, ori_domain_12_ssim_distance]
            ], 

        [
            [[source_image], ori_image_dict[0]],
            [[source_image], ori_image_dict[1]],
            [[source_image], ori_image_dict[2]],
            [transfer_images_dict[0], ori_image_dict[0]],
            [transfer_images_dict[1], ori_image_dict[1]],
            [transfer_images_dict[2], ori_image_dict[2]],
            [ori_image_dict[0], ori_image_dict[0]],
            [ori_image_dict[1], ori_image_dict[1]],
            [ori_image_dict[2], ori_image_dict[2]],
            [transfer_images_dict[0], transfer_images_dict[1]], 
            [transfer_images_dict[0], transfer_images_dict[2]], 
            [transfer_images_dict[1], transfer_images_dict[2]],
            [ori_image_dict[0], ori_image_dict[1]], 
            [ori_image_dict[0], ori_image_dict[2]], 
            [ori_image_dict[1], ori_image_dict[2]]
            ]
        ):
        for i, image_i in enumerate(images[0]):
            image_i = image_i.astype(np.uint8)
            for j, image_j in enumerate(images[1]):
                image_j = image_j.astype(np.uint8)
                ssim_distance = ssim(image_i, image_j, channel_axis=2)
                result_arrs[1][i, j] = ssim_distance
                psnr_distance = psnr(image_i, image_j)
                result_arrs[0][i, j] = psnr_distance

    print("Mean PSNR between domain 0 and domain 1: ", np.mean(domain_01_psnr_distance))
    print("Mean PSNR between domain 0 and domain 2: ", np.mean(domain_02_psnr_distance))
    print("Mean PSNR between domain 1 and domain 2: ", np.mean(domain_12_psnr_distance))
    print("Mean PSNR between ori domain 0 and domain 1: ", np.mean(ori_domain_01_psnr_distance))
    print("Mean PSNR between ori domain 0 and domain 2: ", np.mean(ori_domain_02_psnr_distance))
    print("Mean PSNR between ori domain 1 and domain 2: ", np.mean(ori_domain_12_psnr_distance))

    print("Max PSNR between domain 0 and domain 1: ", np.max(domain_01_psnr_distance))
    print("Max PSNR between domain 0 and domain 2: ", np.max(domain_02_psnr_distance))
    print("Max PSNR between domain 1 and domain 2: ", np.max(domain_12_psnr_distance))
    print("Max PSNR between ori domain 0 and domain 1: ", np.max(ori_domain_01_psnr_distance))
    print("Max PSNR between ori domain 0 and domain 2: ", np.max(ori_domain_02_psnr_distance))
    print("Max PSNR between ori domain 1 and domain 2: ", np.max(ori_domain_12_psnr_distance))

    print("Mean PSNR between source and domain 0: ", np.mean(source_domain_0_psnr_distance))
    print("Mean PSNR between source and domain 1: ", np.mean(source_domain_1_psnr_distance))
    print("Mean PSNR between source and domain 2: ", np.mean(source_domain_2_psnr_distance))
    print("Mean PSNR between ori domain 0 and transfered domain 0: ", np.mean(domain_0_transfer_psnr_distance))
    print("Mean PSNR between ori domain 1 and transfered domain 1: ", np.mean(domain_1_transfer_psnr_distance))
    print("Mean PSNR between ori domain 2 and transfered domain 2: ", np.mean(domain_2_transfer_psnr_distance))
    print("Mean PSNR between ori domain 0 and self domain 0: ", np.mean([i for i in domain_0_self_psnr_distance.flatten() if i != np.inf]))
    print("Mean PSNR between ori domain 1 and self domain 1: ", np.mean([i for i in domain_1_self_psnr_distance.flatten() if i != np.inf]))
    print("Mean PSNR between ori domain 2 and self domain 2: ", np.mean([i for i in domain_2_self_psnr_distance.flatten() if i != np.inf]))

    print("===============================================")

    print("Mean SSIM between source and domain 0: ", np.mean(source_domain_0_ssim_distance))
    print("Mean SSIM between source and domain 1: ", np.mean(source_domain_1_ssim_distance))
    print("Mean SSIM between source and domain 2: ", np.mean(source_domain_2_ssim_distance))
    print("Mean SSIM between ori domain 0 and transfered domain 0: ", np.mean(domain_0_transfer_ssim_distance))
    print("Mean SSIM between ori domain 1 and transfered domain 1: ", np.mean(domain_1_transfer_ssim_distance))
    print("Mean SSIM between ori domain 2 and transfered domain 2: ", np.mean(domain_2_transfer_ssim_distance))
    print("Mean SSIM between ori domain 0 and self domain 0: ", np.mean([i for i in domain_0_self_ssim_distance.flatten() if i != np.inf]))
    print("Mean SSIM between ori domain 1 and self domain 1: ", np.mean([i for i in domain_1_self_ssim_distance.flatten() if i != np.inf]))
    print("Mean SSIM between ori domain 2 and self domain 2: ", np.mean([i for i in domain_2_self_ssim_distance.flatten() if i != np.inf]))

    print("Mean SSIM between domain 0 and domain 1: ", np.mean(domain_01_ssim_distance))
    print("Mean SSIM between domain 0 and domain 2: ", np.mean(domain_02_ssim_distance))
    print("Mean SSIM between domain 1 and domain 2: ", np.mean(domain_12_ssim_distance))
    pass