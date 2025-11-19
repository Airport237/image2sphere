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

    Args:
        cfg (dict): a dictionary for experiment config.
        split (string, optional): 'train', 'val', or 'test'. Defaults to 'train'.
        transforms (callable, optional): a set of Albumentations transformation functions for images.
        target_generators (callable, optional): a function to generate target labels.
    """
    def __init__(self,
                 cfg,
                 split='train',
                 transforms=None,
                 target_generators=None
    ):
        self.root        = '/content/speedplus_data'
        self.is_train    = split == 'train'
        self.split = split
        self.image_size  = [1900, 1200] # Original image size
        self.input_size  = [768, 512] # CNN input size
        #self.output_size = [int(s / cfg.DATASET.OUTPUT_SIZE[0]) for s in self.input_size]
        # List of heads
        #self.load_masks  = True if 'segmentation' in self.head_names else False

        # Folder names
        # TODO: Make the naming automatic based on input image size or specify it at CFG
        self.imagefolder = 'images_768x512_RGB'
        self.maskfolder  = 'masks_192x128'
        self.stylefolder = 'styles_768x512_RGB'

        # Load CSV & determine image domain
        self.csv, self.domain = self._load_csv('/content/speedplus_data/sunlamp/sunlamp/labels/test.csv')

        # Image transforms
        self.transforms = transforms

        images = []
        rotQ = []
        trans = []
        rotM = []

        for index in range(len(self.csv)):
            folder = self.imagefolder

            rot = np.array(self.csv.iloc[index, 1:5], dtype=np.float32) # [qw, qx, qy, qz]
            rotQ.append(rot)
            trans.append(np.array(self.csv.iloc[index, 5:8], dtype=np.float32))
            rotM.append(self.quat2dcm(rot))
            imgpath = join(self.root, self.domain, self.domain, folder, self.csv.iloc[index, 0])
            data = cv2.imread(imgpath, cv2.IMREAD_COLOR)

            images.append(data)

        self.data = {'img': images,
            'rotQ': rotQ,
            'rotM': rotM,
            'trans': trans
            #'cls': torch.from_numpy(data['cat_ids']).unsqueeze(-1).long()
        }

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index):
        assert index < len(self), 'Index range error'

        #------------ Read image
        image = self.data['img'][index]

        #------------ Read all annotations
        anno = self.data

        #------------- Read mask
        # if self.load_labels and self.load_masks:
        #     mask = self._load_mask(index)
        #     anno['mask'] = to_tensor(mask)

        #------------ Image data transform
        # if self.load_labels:
        #     transform_kwargs = {'image': image,
        #                         'bboxes': [anno['boundingbox']],
        #                         'class_labels': ['tango']}
        # else:
        #     transform_kwargs = {'image': image}

        # if self.transforms is not None:
        #     transformed = self.transforms(**transform_kwargs)
        #
        #     # Clean up
        #     image = transformed['image']
        #     if self.load_labels:
        #         anno['boundingbox'] = np.array(transformed['bboxes'][0], dtype=np.float32)

        # # Return just transformed image if not returning labels
        # if not self.load_labels:
        #     return image

        # # Bounding box in [0, 1] -> convert to pixels
        # anno['boundingbox'] *= np.array(
        #     [self.input_size[0], self.input_size[1], self.input_size[0], self.input_size[1]],
        #     dtype=np.float32
        # )

        #------------ Generate targets
        targets = {'domain':         self.split,
                   'quaternion':     torch.from_numpy(anno['rotQ']),
                   'rotationmatrix': torch.from_numpy(anno['rotM']),
                   'translation':    torch.from_numpy(anno['trans'])}

        # # Additional targets if training
        # if self.is_train:
        #     if self.load_masks:
        #         targets['mask'] = anno['mask']
        #
        #     for i, h in enumerate(self.head_names):
        #         if h == 'heatmap':
        #             heatmap = self.target_generators[i](anno['keypoints']).astype(np.float32)
        #             targets['heatmap'] = torch.from_numpy(heatmap)
        #         elif h == 'efficientpose' or h == 'segmentation':
        #             pass
        #         else:
        #             raise ValueError(f'{h} is not a valid head name')

        return image, targets

    # def _load_csv(self, split='train'):
    #     """ Load CSV content into pandas.DataFrame """
    #
    #     # Current domain
    #     domain = split
    #
    #
    #     # Read CSV file to pandas
    #     csv = pd.read_csv(join(self.root, split, split, 'labels','test.csv'), header=None)
    #
    #     return csv, domain

    # def _load_image(self, index, folder=None):
    #     """ Read image of given index from a folder, if specified """
    #
    #     # Overwrite image folder if not provided
    #     if folder is None:
    #         folder = self.imagefolder
    #
    #     # Read
    #     imgpath = join(self.root, self.domain, self.domain, folder, self.csv.iloc[index, 0])
    #     data    = cv2.imread(imgpath, cv2.IMREAD_COLOR)
    #     # data    = cv2.cvtColor(data, cv2.BGR2RGB) # Uncomment if actual RGB color image
    #     return data
    #
    # def _load_mask(self, index):
    #     """ Read mask image """
    #
    #     imgpath = join(self.root, self.domain, self.maskfolder, self.csv.iloc[index, 0])
    #     data    = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
    #     # data    = cv2.resize(data, self.output_size) # Uncomment if resizing images in Dataset class
    #
    #     # Clean up any intermediate values
    #     data[data >  128] = 255
    #     data[data <= 128] = 0
    #
    #     return data[:,:,None]

    def quat2dcm(self, q):
        """ Computing direction cosine matrix from quaternion, adapted from PyNav. """
        # normalizing quaternion
        q = q / np.linalg.norm(q)

        q0 = q[0]
        q1 = q[1]
        q2 = q[2]
        q3 = q[3]

        dcm = np.zeros((3, 3), dtype=np.float32)

        dcm[0, 0] = 2 * q0 ** 2 - 1 + 2 * q1 ** 2
        dcm[1, 1] = 2 * q0 ** 2 - 1 + 2 * q2 ** 2
        dcm[2, 2] = 2 * q0 ** 2 - 1 + 2 * q3 ** 2

        dcm[0, 1] = 2 * q1 * q2 + 2 * q0 * q3
        dcm[0, 2] = 2 * q1 * q3 - 2 * q0 * q2

        dcm[1, 0] = 2 * q1 * q2 - 2 * q0 * q3
        dcm[1, 2] = 2 * q2 * q3 + 2 * q0 * q1

        dcm[2, 0] = 2 * q1 * q3 + 2 * q0 * q2
        dcm[2, 1] = 2 * q2 * q3 - 2 * q0 * q1

        return dcm
