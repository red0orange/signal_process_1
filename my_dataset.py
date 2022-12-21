import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class ProDataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):
        super(ProDataset, self).__init__()
        # self.flag = "training" if train else ["Domain1", "Domain2", "Domain3"]
        self.flag = "Enhance_training" if train else ["Domain1", "Domain2", "Domain3"]

        if not isinstance(self.flag, list):
            data_root = os.path.join(root, "SegmentationData", self.flag)
            assert os.path.exists(data_root), f"path '{data_root}' does not exists."
            self.transforms = transforms
            img_names = [i for i in os.listdir(os.path.join(data_root, "data")) if (i.endswith(".bmp") or i.endswith(".jpg"))]
            label_names = [i for i in os.listdir(os.path.join(data_root, "label")) if (i.endswith(".bmp") or i.endswith(".jpg") or i.endswith(".png"))]
            mask_names = [i for i in os.listdir(os.path.join(data_root, "mask")) if (i.endswith(".bmp") or i.endswith(".jpg") or i.endswith(".png"))]

            self.img_list = [os.path.join(data_root, "data", i) for i in img_names]
            self.img_list = sorted(self.img_list, key=lambda i: os.path.basename(i).rsplit(".", maxsplit=1)[0])
            self.label = [os.path.join(data_root, "label", i) for i in label_names]
            self.label = sorted(self.label, key=lambda i: os.path.basename(i).rsplit(".", maxsplit=1)[0])
            self.roi_mask = [os.path.join(data_root, "mask", i) for i in mask_names]
            self.roi_mask = sorted(self.roi_mask, key=lambda i: os.path.basename(i).rsplit(".", maxsplit=1)[0])
        else:
            self.img_list = []
            self.roi_mask = []
            self.label = []
            for flag in self.flag:
                data_root = os.path.join(root, "SegmentationData", flag)
                assert os.path.exists(data_root), f"path '{data_root}' does not exists."
                self.transforms = transforms
                img_names = [i for i in os.listdir(os.path.join(data_root, "data")) if (i.endswith(".bmp") or i.endswith(".jpg"))]
                label_names = [i for i in os.listdir(os.path.join(data_root, "label")) if (i.endswith(".bmp") or i.endswith(".jpg") or i.endswith(".png"))]
                mask_names = [i for i in os.listdir(os.path.join(data_root, "mask")) if (i.endswith(".bmp") or i.endswith(".jpg") or i.endswith(".png"))]

                self.img_list.extend([os.path.join(data_root, "data", i) for i in img_names])
                self.label.extend([os.path.join(data_root, "label", i) for i in label_names])
                self.roi_mask.extend([os.path.join(data_root, "mask", i) for i in mask_names])
            self.img_list = sorted(self.img_list, key=lambda i: os.path.basename(i).rsplit(".", maxsplit=1)[0])
            self.label = sorted(self.label, key=lambda i: os.path.basename(i).rsplit(".", maxsplit=1)[0])
            self.roi_mask = sorted(self.roi_mask, key=lambda i: os.path.basename(i).rsplit(".", maxsplit=1)[0])

        # check files
        for i in self.label:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")

        # self.roi_mask = [os.path.join(data_root, "mask", i.split("_")[0] + f"_{self.flag}_mask.gif")
        #                  for i in img_names]
        # # check files
        # for i in self.roi_mask:
        #     if os.path.exists(i) is False:
        #         raise FileNotFoundError(f"file {i} does not exists.")

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB')
        manual = Image.open(self.label[idx]).convert('L')
        manual = np.array(manual) / 255
        roi_mask = Image.open(self.roi_mask[idx])
        roi_mask = 255 - np.array(roi_mask)
        # roi_mask[np.bitwise_and(roi_mask != 0, roi_mask != 255)] = 255
        mask = np.clip(manual + roi_mask, a_min=0, a_max=255)

        # 这里转回PIL的原因是，transforms中是对PIL数据进行处理
        mask = Image.fromarray(mask)

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs

