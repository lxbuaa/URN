import time

import easydict
import random
import cv2
import numpy as np
import albumentations as alb
import albumentations.pytorch as albp
import os.path as osp
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch
from torchvision.utils import make_grid


def get_grid_img(img, gt, pred, k=0.5):
    gt_255 = gt * 255.0
    print(gt_255)
    gt_255 = torch.repeat_interleave(gt_255, repeats=3, dim=0)
    pred_255 = (pred >= k) * 255.0
    print(pred_255)
    pred_255 = torch.repeat_interleave(pred_255, repeats=3, dim=0)
    show_list = [img, gt_255, pred_255]
    grid_img = make_grid(show_list, nrow=3, padding=20, normalize=True,
                         scale_each=True, pad_value=1)
    np_grid_img = grid_img.detach().cpu().numpy()
    np_grid_img = np_grid_img.transpose(1, 2, 0)
    return np_grid_img


def get_img_and_mask(img_dir, mask_dir, cls_index, keep_scale=False):
    """
    :param cls_index: 1 = spliced, 0 = pristine
    :param img_dir: Path of image file
    :param mask_dir: Path of mask file
    :return: a dict with keys 'img', 'mask'
    """
    img = cv2.imread(img_dir)
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if mask_dir == 'None':
            try:
                if cls_index == 0:
                    mask = np.zeros((img.shape[0], img.shape[1]))
                else:
                    mask = np.ones((img.shape[0], img.shape[1])) * 255
            except AttributeError:
                print(img_dir, mask_dir)
                if cls_index == 0:
                    mask = np.zeros((img.shape[0], img.shape[1]))
                else:
                    mask = np.ones((img.shape[0], img.shape[1])) * 255
        else:
            mask = cv2.imread(mask_dir, cv2.IMREAD_GRAYSCALE)

    except ValueError:
        print(img_dir, mask_dir)
    try:
        return {
            "img": np.array(img),
            "mask": np.array(mask)
        }
    except UnboundLocalError:
        print(img_dir)
        print(mask_dir)


class BioBaseDataset(torch.utils.data.Dataset):
    """
    An abstract dataset
    """

    def __init__(self,
                 config,
                 txt_file=None,
                 img_size=256,
                 split='train',
                 keep_scale=False,
                 ):
        """
        :param config:
        :param txt_file:
        :param img_size:
        :param split: train or test
        """
        self.txt_file = txt_file
        self.split = split
        self.img_list = []
        self.mask_list = []
        self.cls_list = []
        self.config = config
        self.read_txt()
        if self.split == 'train':
            """training data augment"""
            self.transforms = alb.Compose(
                # [
                #     # alb.RandomResizedCrop(img_size, img_size, interpolation=cv2.INTER_CUBIC, p=0.5),
                #     alb.Resize(img_size, img_size, interpolation=cv2.INTER_NEAREST),
                #     # alb.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=0.1, p=1),
                #     # alb.RandomRotate90(p=0.5),
                #     alb.HorizontalFlip(p=0.5),
                #     # alb.VerticalFlip(p=0.5),
                #     # alb.Blur(p=0.3),
                #     # alb.ImageCompression(quality_lower=70, quality_upper=100, p=0.1),
                #     # alb.GaussianBlur(p=0.1),
                #     # RandomCopyMove(p=0.1),
                #     # RandomInpainting(p=0.1),
                #     alb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                #     albp.transforms.ToTensorV2(),
                # ]
                [
                    alb.Resize(img_size, img_size, interpolation=cv2.INTER_NEAREST),
                    alb.RandomRotate90(p=0.5),
                    alb.HorizontalFlip(p=0.5),
                    alb.ImageCompression(quality_lower=20, quality_upper=100, p=0.1),
                    alb.GaussianBlur(blur_limit=13, p=0.1),
                    alb.GaussNoise(var_limit=(10, 80), p=0.1),
                    alb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    albp.transforms.ToTensorV2(),
                ]
            )
        else:
            """testing data augment"""
            function_list = []
            function_list.extend([
                alb.Resize(img_size, img_size, interpolation=cv2.INTER_NEAREST),
                alb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                albp.transforms.ToTensorV2(),
            ])
            self.transforms = alb.Compose(
                function_list
            )

        self.original_mask_transform = alb.Compose([])


    def read_txt(self):
        if self.txt_file is None:
            print("Initializing an empty dataset!")
            return
        with open(self.txt_file, 'r') as f:  # Image path [space] mask path or None [space] 0 or 1“
            line = f.readline()
            while line:
                line = line.replace('\n', '').replace('\r', '')
                try:
                    img_dir, mask_dir = self.get_true_dir(line.split(' ')[0], line.split(' ')[1])
                except IndexError:
                    print(line, '!!')
                    img_dir, mask_dir = self.get_true_dir(line.split(' ')[0], line.split(' ')[1])

                self.img_list.append(img_dir)
                self.mask_list.append(mask_dir)
                cls_gt = line.split(' ')[2]
                self.cls_list.append(1 if int(cls_gt) > 0 else 0)
                line = f.readline()

    def get_true_dir(self, img_dir, mask_dir):
        """
        :param img_dir:
        :param mask_dir:
        :return: no return. Just put it in the list.
        """
        self.img_list.append(img_dir)
        self.mask_list.append(mask_dir)

    def __len__(self):
        return len(self.img_list)

    def read_img_mask(self, index):
        this_img = self.img_list[index]
        this_mask = self.mask_list[index]
        this_cls = int(self.cls_list[index])
        x = get_img_and_mask(this_img, this_mask, this_cls, self.keep_scale)  # 预处理图像和mask
        return x

    # @torchsnooper.snoop()
    def __getitem__(self, index):
        """
        get data by the index
        :param index: the index of the item we need to get
        :return: easydict
        """
        x = self.read_img_mask(index)

        """进行数据增强"""
        try:
            aug = self.transforms(image=x['img'], mask=x['mask'])
        except:
            print(x['img'])
            print(x['mask'])
            print(self.img_list[index], self.mask_list[index])
            raise ValueError
        img = aug['image']
        mask = aug['mask'] / 255.0

        res_dict = easydict.EasyDict({
            "img": img,
            "mask": mask,
            "cls": torch.max(mask),
            "name": self.img_list[index].split('/')[-1]
        })
        if self.txt_file.find("RSIIL") != -1:
            res_dict['name'] = self.img_list[index].split('/')[-2] + res_dict['name']
        elif self.txt_file.find("HandforsV2") != -1:
            res_dict['name'] = self.img_list[index].split('/')[-2] + res_dict['name']

        if self.split == 'test':
            res_dict['original_mask'] = x['mask'] / 255.0

        return res_dict


class BioDataset(BioBaseDataset):
    def get_true_dir(self, img_dir, mask_dir):
        img_dir = osp.join(self.config.bio_dataset_dir, img_dir)
        if mask_dir != 'None':
            mask_dir = osp.join(self.config.bio_dataset_dir, mask_dir)
        return img_dir, mask_dir