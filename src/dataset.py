from os.path import join
from random import random
from typing import Optional, List, Callable
import os
import glob

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image

class ModelNet10Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_path: str,
                 train: bool,
                 limited: bool = False,
                ):
        name = f"modelnet10_{'train' if train else 'test'}.npz"
        if limited and train:
            name = name.replace('_', '_limited_')
        path = os.path.join(dataset_path, "modelnet10", name)
        data = np.load(path)

        self.data = {
            'img' : torch.from_numpy(data['imgs']).permute(0, 3, 1, 2),
            'rot' : torch.from_numpy(data['rots']),
            'cls' : torch.from_numpy(data['cat_ids']).unsqueeze(-1).long(),
        }

        self.num_classes = 10
        self.class_names = ('bathtub', 'bed', 'chair', 'desk', 'dresser',
                            'monitor', 'night_stand', 'sofa', 'table', 'toilet')

    def __getitem__(self, index):
        img = self.data['img'][index].to(torch.float32) / 255.

        if img.shape[0] != 3:
            img = img.expand(3,-1,-1)

        class_index = self.data['cls'][index]

        rot = self.data['rot'][index]

        return dict(img=img, cls=class_index, rot=rot)

    @property
    def img_shape(self):
        return (3, 224, 224)

    def __len__(self):
        return len(self.data['img'])


class SymsolDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_path: str,
                 train: bool,
                 set_number: int=1,
                 num_views: int=None,
                ):
        self.mode = 'train' if train else 'test'
        self.path = os.path.join(dataset_path, "symsol", self.mode)
        rotations_data = np.load(os.path.join(self.path, 'rotations.npz'))
        self.class_names = {
            1 : ('tet', 'cube', 'icosa', 'cone', 'cyl'),
            2 : ('sphereX',),#, 'cylO', 'sphereX'),
            3 : ('cylO',),#, 'cylO', 'sphereX'),
            4 : ('tetX',),#, 'cylO', 'sphereX'),
        }[set_number]
        self.num_classes = len(self.class_names)

        self.rotations_data = [rotations_data[c][:num_views] for c in self.class_names]
        self.indexers = np.cumsum([len(v) for v in self.rotations_data])

    def __getitem__(self, index):
        cls_ind = np.argmax(index < self.indexers)
        if cls_ind > 0:
            index = index - self.indexers[cls_ind-1]

        rot = self.rotations_data[cls_ind][index]
        # randomly sample one of the valid rotation labels
        rot = rot[np.random.randint(len(rot))]
        rot = torch.from_numpy(rot)

        im_path = os.path.join(self.path, 'images',
                               f'{self.class_names[cls_ind]}_{str(index).zfill(5)}.png')
        img = np.array(Image.open(im_path))
        img = torch.from_numpy(img).to(torch.float32) / 255.
        img = img.permute(2, 0, 1)

        class_index = torch.tensor((cls_ind,), dtype=torch.long)

        return dict(img=img, cls=class_index, rot=rot)

    def __len__(self):
        return self.indexers[-1]

    @property
    def img_shape(self):
        return (3, 224, 224)






class SPEEDPLUSDataset(torch.utils.data.Dataset):
    """ PyTorch Dataset class for SPEED+
    """
    def __init__(self,
                 root: str,
                 split: str = 'train',
                 transforms=None,
    ):
        self.root        = root                # e.g. args.dataset_path + '/speedplus_data'
        self.split       = split               # 'train' or 'test'
        self.is_train    = split == 'train'
        self.image_size  = [1900, 1200]        # original
        self.input_size  = [768, 512]          # CNN input size (W, H)

        self.imagefolder = 'images_768x512_RGB'
        self.maskfolder  = 'masks_192x128'
        self.stylefolder = 'styles_768x512_RGB'

        # Load CSV & determine image domain
        self.csv, self.domain = self._load_csv(
            os.path.join(self.root, 'sunlamp', 'sunlamp', 'labels', 'test.csv')
        )

        self.transforms = transforms

        # store per-sample data
        self.imgs   = []
        self.rots   = []   # rotation matrices 3x3
        self.trans  = []   # translation vectors (3,)

        for index in range(len(self.csv)):
            rot_q = np.array(self.csv.iloc[index, 1:5], dtype=np.float32)   # [qw, qx, qy, qz]
            t     = np.array(self.csv.iloc[index, 5:8], dtype=np.float32)   # (3,)

            R = self.quat2dcm(rot_q)                                        # (3, 3)

            imgpath = join(self.root, self.domain, self.domain,
                           self.imagefolder, self.csv.iloc[index, 0])
            img_bgr = cv2.imread(imgpath, cv2.IMREAD_COLOR)                 # HxWx3 BGR

            # optional: resize to input_size
            img_bgr = cv2.resize(img_bgr, tuple(self.input_size))           # (H=512, W=768)

            img = torch.from_numpy(img_bgr).to(torch.float32) / 255.0       # HxWx3
            img = img.permute(2, 0, 1)                                      # 3xHxW

            self.imgs.append(img)
            self.rots.append(torch.from_numpy(R))                           # 3x3
            self.trans.append(torch.from_numpy(t))                          # 3,

        # make it look like other datasets
        self.num_classes = 1
        self.class_names = ('tango',)   # or any name, you only have 1 class

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img   = self.imgs[index]
        rot   = self.rots[index]        # 3x3
        trans = self.trans[index]       # 3,

        # dummy single class; shape (1,) to match others
        cls = torch.zeros(1, dtype=torch.long)

        return dict(img=img, cls=cls, rot=rot, trans=trans)

    @property
    def img_shape(self):
        # (C, H, W)
        return (3, self.input_size[1], self.input_size[0])

    def _load_csv(self, csv_path):
        csv = pd.read_csv(csv_path, header=None)
        # in your original code, domain == split; keep it simple
        domain = self.split
        return csv, domain

    def quat2dcm(self, q):
        """ Computing direction cosine matrix from quaternion, adapted from PyNav. """
        q = q / np.linalg.norm(q)

        q0, q1, q2, q3 = q
        dcm = np.zeros((3, 3), dtype=np.float32)

        dcm[0, 0] = 2 * q0**2 - 1 + 2 * q1**2
        dcm[1, 1] = 2 * q0**2 - 1 + 2 * q2**2
        dcm[2, 2] = 2 * q0**2 - 1 + 2 * q3**2

        dcm[0, 1] = 2 * q1 * q2 + 2 * q0 * q3
        dcm[0, 2] = 2 * q1 * q3 - 2 * q0 * q2
        dcm[1, 0] = 2 * q1 * q2 - 2 * q0 * q3
        dcm[1, 2] = 2 * q2 * q3 + 2 * q0 * q1
        dcm[2, 0] = 2 * q1 * q3 + 2 * q0 * q2
        dcm[2, 1] = 2 * q2 * q3 - 2 * q0 * q1

        return dcm

