import os

import torch

from unet import UNet
from train_utils import evaluate
from my_dataset import ProDataset
import transforms as T


class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)
        max_size = int(1.2 * base_size)

        trans = [T.RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.extend([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.Resize(480),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    base_size = 565
    crop_size = 480

    if train:
        return SegmentationPresetTrain(base_size, crop_size, mean=mean, std=std)
    else:
        return SegmentationPresetEval(mean=mean, std=std)


def create_model(num_classes):
    model = UNet(in_channels=3, num_classes=num_classes, base_c=32)
    return model


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    # segmentation nun_classes + background
    num_classes = args.num_classes + 1

    # using compute_mean_std.py
    mean=(0.485, 0.456, 0.406)
    std=(0.229, 0.224, 0.225)

    val_dataset = ProDataset(args.data_path,
                               train=False,
                               transforms=get_transform(train=False, mean=mean, std=std))

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=num_classes)
    model.load_state_dict(torch.load(args.weights_path, map_location='cpu')['model'])
    model.to(device)

    confmat, dice = evaluate(model, val_loader, device=device, num_classes=num_classes)
    val_info = str(confmat)
    print(val_info)
    print(f"dice coefficient: {dice:.3f}")
    pass


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch unet test")

    parser.add_argument("--data-path", default="./", help="DRIVE root")
    # parser.add_argument("--weights-path", default="/home/dehao/github_projects/pro/results/generailzation_epoch_8_best_model.pth")
    # parser.add_argument("--weights-path", default="/home/dehao/github_projects/pro/results/epoch_50_best_model.pth")
    # parser.add_argument("--weights-path", default="/home/dehao/github_projects/pro/results/best_model_withmask_50.pth")
    # parser.add_argument("--weights-path", default="/home/dehao/github_projects/pro/results/best_model_withoutmask_50.pth")
    # parser.add_argument("--weights-path", default="/home/dehao/github_projects/pro/results/best_model_withoutmask_50_10.pth")
    parser.add_argument("--weights-path", default="/home/dehao/github_projects/pro/results/best_model_with_mask.pth")
    # exclude background
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=12, type=int)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
