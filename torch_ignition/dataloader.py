#!/usr/bin/env python

import os
import numpy as np
import random
from PIL import Image

import torch
from torchvision import transforms, datasets


class CustomDataloader(torch.utils.data.Dataset):

    def __init__(self,
                 dataset: list,
                 input_shape: list = [3, 224, 224],
                 batch_size: int = 1,
                 transform = None,
                 target_transform = None,
                 shuffle: bool = False):
        super(CustomDataloader, self).__init__()

        assert batch_size > 0
        assert len(input_shape) == 3
        
        self._transform = transform
        self._target_transform = target_transform
        self._input_shape = input_shape
        self._batch_size = batch_size
        self._shuffle = shuffle

        self.dataset = dataset
        self.indices_cache = np.arange(0, len(self.dataset))

    def __len__(self):
        return len(self.dataset.images) // self._batch_size

    def __getitem__(self, index: int):
        if torch.is_tensor(index):
            index = index.tolist()

        if self._shuffle:
            np.random.seed(index)
            np.random.shuffle(self.indices_cache)

        batch_images = torch.zeros(self._batch_size,
                                   *self._input_shape)
        batch_masks = torch.zeros(self._batch_size,
                                  *self._input_shape[1:]).long()

        dsize = len(self.dataset)
        st = index * self._batch_size
        ed = st + self._batch_size
        if dsize < ed:
            st -= (ed - dsize - 1)
            ed = dsize - 1
        
        for k, i in enumerate(self.indices_cache[st:ed]):
            im, mk = self.load(index=i)
            batch_images[k] = im
            batch_masks[k] = mk

        return batch_images, batch_masks

    def load(self, index: int):
        im, mk = self.dataset[index]
        
        if np.random.choice([0, 1]):
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
            mk = mk.transpose(Image.FLIP_LEFT_RIGHT)

        seed = np.random.randint(2147483647)
        random.seed(seed)
        if self._transform is not None:
            im = self._transform(im)

        random.seed(seed)
        if self._target_transform is not None:
            max_v = np.max(mk)
            mk = self._target_transform(mk)
            mk *= max_v
            mk = mk.type(torch.LongTensor)
        return im, mk


class Dataloader(object):

    def __init__(self,
                 dataset_dir: str,
                 input_shape: list = [3, 224, 224],
                 batch_size: int = 1):
    
        assert len(input_shape) == 3
        assert os.path.isdir(dataset_dir)

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        transform_train = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.RandomResizedCrop(input_shape[1:]),
            transforms.ColorJitter([0.5, 2], [0.5, 2], 0.5, 0.5),
            # transforms.RandomAffine(degrees=180, scale=(0.2, 2.0)),
            # transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        
        interpolation = Image.NEAREST
        transform_train_target = transforms.Compose([
            transforms.Resize([256, 256],
                              interpolation=interpolation),
            transforms.RandomResizedCrop(input_shape[1:],
                                         interpolation=interpolation),
            # transforms.RandomAffine(degrees=180),
            # transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])
        
        transform_val = transforms.Compose([
            transforms.Resize(input_shape[1:]),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        transform_val_target = transforms.Compose([
            transforms.Resize(input_shape[1:],
                              interpolation=interpolation),
            transforms.ToTensor(),
        ])

        worker = 8 if os.cpu_count() > 8 else os.cpu_count()

        download = False
        train_ds = datasets.VOCSegmentation(
            root=dataset_dir, image_set='train', download=download)
        
        self._train_dl = CustomDataloader(
            train_ds, batch_size=batch_size,
            input_shape=input_shape,
            transform=transform_train,
            target_transform=transform_train_target,
            shuffle=True)

        val_ds = datasets.VOCSegmentation(
            root=dataset_dir, image_set='val', download=download)
        self._val_dl = CustomDataloader(
            val_ds, batch_size=batch_size,
            input_shape=input_shape,
            transform=transform_val,
            target_transform=transform_val_target,
            shuffle=False)

    @property
    def train_data(self):
        return self._train_dl

    @property
    def val_data(self):
        return self._val_dl


if __name__ == '__main__':

    d = Dataloader('/workspace/datasets/', input_shape=[3, 224, 224], batch_size=128)

    for i in d.train_data:
        print(len(i))
        break
    
    import IPython
    IPython.embed()
