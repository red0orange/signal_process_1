import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def generate_filter_matrix(image_shape, thres_f, filter_type="lowpass"):
    if filter_type == "lowpass":
        filter_matrix = np.zeros(image_shape, dtype=np.uint8)
        fill_value = 1
    elif filter_type == "highpass":
        filter_matrix = np.ones(image_shape, dtype=np.uint8)
        fill_value = 0
    else:
        raise BaseException("Error Type")
    if len(image_shape) == 3:
        fill_value = [fill_value] * image_shape[2]

    center_point = (image_shape[0] // 2, image_shape[1] // 2)
    assert (thres_f < center_point[0]) and (thres_f < center_point[1])

    cv2.rectangle(filter_matrix, (center_point[0] - thres_f, center_point[1] - thres_f), (center_point[0] + thres_f, center_point[1] + thres_f), color=fill_value, thickness=-1)

    return filter_matrix

def real2complex(amplitude, phase):
    return np.vectorize(complex)(amplitude * np.cos(phase), amplitude * np.sin(phase))

def image_domain_transfer(source_image, target_domain_image, L, lam, reshape_shape=[384, 384], return_f=False):
    # source_image = cv2.imread(source_image_path)[:, :, ::-1]
    source_image = cv2.resize(source_image, reshape_shape)
    source_image_f = np.fft.fft2(source_image, axes=[0, 1])
    source_image_phase, source_image_amplitude = np.angle(source_image_f), np.abs(source_image_f)
    source_image_fshift = np.fft.fftshift(source_image_f, axes=[0, 1])
    source_image_fshift_phase = np.fft.fftshift(source_image_phase, axes=[0, 1])
    source_image_fshift_amplitude = np.fft.fftshift(source_image_amplitude, axes=[0, 1])
    source_image_view_f = np.clip(np.log(source_image_fshift_amplitude) / np.max(np.log(source_image_fshift_amplitude)), 0, 1)

    # target_domain_image = cv2.imread(target_domain_image_path)[:, :, ::-1]
    target_domain_image = cv2.resize(target_domain_image, reshape_shape)
    target_domain_image_f = np.fft.fft2(target_domain_image, axes=[0, 1])
    target_domain_image_phase, target_domain_image_amplitude = np.angle(target_domain_image_f), np.abs(target_domain_image_f)
    target_domain_image_fshift = np.fft.fftshift(target_domain_image_f, axes=[0, 1])
    target_domain_image_fshift_phase = np.fft.fftshift(target_domain_image_phase, axes=[0, 1])
    target_domain_image_fshift_amplitude = np.fft.fftshift(target_domain_image_amplitude, axes=[0, 1])
    target_domain_image_view_f = np.clip(np.log(target_domain_image_fshift_amplitude) / np.max(np.log(target_domain_image_fshift_amplitude)), 0, 1)

    # core part
    transfer_image_fshift_amplitude = source_image_fshift_amplitude.copy()
    h, w, _ = transfer_image_fshift_amplitude.shape
    b = (np.floor(np.amin((h,w))*L)).astype(int)
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)
    h1 = c_h - b
    h2 = c_h + b + 1
    w1 = c_w - b
    w2 = c_w + b + 1
    transfer_image_fshift_amplitude[h1:h2, w1:w2, :] = transfer_image_fshift_amplitude[h1:h2, w1:w2, :] * lam + target_domain_image_fshift_amplitude[h1:h2, w1:w2, :] * (1 - lam)    # the formula
    transfer_image_view_f = np.clip(np.log(transfer_image_fshift_amplitude) / np.max(np.log(transfer_image_fshift_amplitude)), 0, 1)
    transfer_image_amplitude = np.fft.ifftshift(transfer_image_fshift_amplitude, axes=(0, 1))
    transfer_image_f = transfer_image_amplitude * np.exp(1j * source_image_phase)
    transfer_image = np.fft.ifft2(transfer_image_f, axes=(0, 1))
    transfer_image = np.clip(np.real(transfer_image), 0, 255)
    transfer_image_view = np.clip(transfer_image / 255, 0, 1)

    # plt.subplot(161), plt.imshow(source_image_view_f, cmap='gray'), plt.title("Source Image"), plt.axis("off")
    # plt.subplot(162), plt.imshow(target_domain_image_view_f, cmap='gray'), plt.title("Source Image"), plt.axis("off")
    # plt.subplot(163), plt.imshow(transfer_image_view_f, cmap='gray'), plt.title("Target Domain Image"), plt.axis("off")
    # plt.subplot(164), plt.imshow(source_image / 255), plt.title("source Image"), plt.axis("off")
    # plt.subplot(165), plt.imshow(target_domain_image / 255), plt.title("target Image"), plt.axis("off")
    # plt.subplot(166), plt.imshow(transfer_image_view), plt.title("transfer Image"), plt.axis("off")
    # plt.show()

    if return_f:
        return transfer_image, transfer_image_view_f * 255

    return transfer_image


if __name__ == "__main__":
    resize_shape = (384, 384)
    L = 0.003
    lam = 1 - 0.2

    source_image_path = "/home/red0orange/github_projects/FedDG-ELCFS/demo_samples/fundus_client4.jpg"
    target_domain_image_path = "/home/red0orange/github_projects/FedDG-ELCFS/demo_samples/fundus_client1.png"
    source_image = cv2.imread(source_image_path)[:, :, ::-1]
    target_domain_image = cv2.imread(target_domain_image_path)[:, :, ::-1]

    image_domain_transfer(source_image, target_domain_image, L=L, lam=lam, reshape_shape=resize_shape)
    pass
