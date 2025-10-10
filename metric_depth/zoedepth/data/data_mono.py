# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

# This file is partly inspired from BTS (https://github.com/cleinc/bts/blob/master/pytorch/bts_dataloader.py); author: Jin Han Lee

import itertools
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame
import cv2
import torch
import torch.nn as nn
import torch.utils.data.distributed
from zoedepth.utils.easydict import EasyDict as edict
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from zoedepth.utils.config import change_dataset

from .vkitti import get_vkitti_loader
from .vkitti2 import get_vkitti2_loader

from .preprocess import CropParams, get_white_border, get_black_border

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def _resize_with_aspect_and_pad_top_left_pil(img: Image.Image, target_height: int, target_width: int, resample: int) -> Image.Image:
    """Resize keeping aspect ratio and pad with zeros on bottom-right to reach target size.

    The resized content is placed at the top-left corner (0,0). Padding value is 0.
    """
    orig_w, orig_h = img.size
    if orig_w == 0 or orig_h == 0:
        return Image.new(img.mode, (target_width, target_height), 0)

    scale = min(target_width / float(orig_w), target_height / float(orig_h))
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))

    resized = img.resize((new_w, new_h), resample=resample)

    background = 0 if img.mode in ("I", "I;16", "F", "L") else (0, 0, 0)
    canvas = Image.new(img.mode, (target_width, target_height), color=background)
    canvas.paste(resized, (0, 0))
    return canvas


def resize_and_pad_pil_pair(image: Image.Image, depth: Image.Image, target_height: int, target_width: int):
    """Apply aspect-ratio preserving resize with bottom-right zero padding to image and depth.

    Uses bilinear for RGB image and nearest for depth to avoid interpolation artifacts.
    """
    image_resized = _resize_with_aspect_and_pad_top_left_pil(image, target_height, target_width, Image.BILINEAR)
    depth_resized = _resize_with_aspect_and_pad_top_left_pil(depth, target_height, target_width, Image.NEAREST)
    return image_resized, depth_resized


def preprocessing_transforms(mode, **kwargs):
    return transforms.Compose([
        ToTensor(mode=mode, **kwargs)
    ])


def read_exr_depth_as_pil(path):
    arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # float32 if EXR
    if arr is None:
        raise ValueError("Failed to read EXR. Your OpenCV build may lack OpenEXR support.")
    if arr.ndim == 3:
        arr = arr[..., 0]  # pick first channel if needed
    return Image.fromarray(arr.astype(np.float32), mode='F')


class DepthDataLoader(object):
    def __init__(self, config, mode, device='cpu', transform=None, **kwargs):
        """
        Data loader for depth datasets

        Args:
            config (dict): Config dictionary. Refer to utils/config.py
            mode (str): "train" or "online_eval"
            device (str, optional): Device to load the data on. Defaults to 'cpu'.
            transform (torchvision.transforms, optional): Transform to apply to the data. Defaults to None.
        """

        self.config = config

        if config.dataset == 'vkitti':
            self.data = get_vkitti_loader(
                config.vkitti_root, batch_size=1, num_workers=1)
            return

        if config.dataset == 'vkitti2':
            self.data = get_vkitti2_loader(
                config.vkitti2_root, batch_size=1, num_workers=1)
            return

        img_size = self.config.get("img_size", None)
        img_size = img_size if self.config.get(
            "do_input_resize", False) else None

        if transform is None:
            transform = preprocessing_transforms(mode, size=img_size)
            #* Compose(<zoedepth.data.data_mono.ToTensor object)
            #* Normalization is disabled by default

        if config.dataset == 'allo':
            dataset = ALLODataLoadPreprocess
        elif config.dataset == 'STU-Mix':
            dataset = STUMixDataLoadPreprocess
        else:
            dataset = DataLoadPreprocess

        if mode == 'train':
            self.training_samples = dataset(
                config, mode, transform=transform, device=device)

            #TODO: balance samples between STU and others
            #TODO: ensure at least 1 STU sample is in the batch
            if config.distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                    self.training_samples)
            else:
                self.train_sampler = None

            self.data = DataLoader(self.training_samples,
                                   batch_size=config.batch_size,
                                   shuffle=(self.train_sampler is None),
                                   num_workers=config.num_workers,
                                   pin_memory=(config.num_workers > 0),
                                   persistent_workers=(config.num_workers > 0),
                                   sampler=self.train_sampler)

        elif mode == 'online_eval':
            self.testing_samples = dataset(
                config, mode, transform=transform)
            if config.distributed:  # redundant. here only for readability and to be more explicit
                # Give whole test set to all processes (and report evaluation only on one) regardless
                self.eval_sampler = None
            else:
                self.eval_sampler = None
            self.data = DataLoader(self.testing_samples, 1,
                                   shuffle=kwargs.get("shuffle_test", False),
                                   num_workers=1,
                                   pin_memory=False,
                                   sampler=self.eval_sampler)

        elif mode == 'test':
            self.testing_samples = dataset(
                config, mode, transform=transform)
            self.data = DataLoader(self.testing_samples,
                                   1, shuffle=False, num_workers=1)

        else:
            print(
                'mode should be one of \'train, test, online_eval\'. Got {}'.format(mode))


def repetitive_roundrobin(*iterables):
    """
    cycles through iterables but sample wise
    first yield first sample from first iterable then first sample from second iterable and so on
    then second sample from first iterable then second sample from second iterable and so on

    If one iterable is shorter than the others, it is repeated until all iterables are exhausted
    repetitive_roundrobin('ABC', 'D', 'EF') --> A D E B D F C D E
    """
    # Repetitive roundrobin
    iterables_ = [iter(it) for it in iterables]
    exhausted = [False] * len(iterables)
    while not all(exhausted):
        for i, it in enumerate(iterables_):
            try:
                yield next(it)
            except StopIteration:
                exhausted[i] = True
                iterables_[i] = itertools.cycle(iterables[i])
                # First elements may get repeated if one iterable is shorter than the others
                yield next(iterables_[i])


class RepetitiveRoundRobinDataLoader(object):
    def __init__(self, *dataloaders):
        self.dataloaders = dataloaders

    def __iter__(self):
        return repetitive_roundrobin(*self.dataloaders)

    def __len__(self):
        # First samples get repeated, thats why the plus one
        return len(self.dataloaders) * (max(len(dl) for dl in self.dataloaders) + 1)


class MixedNYUKITTI(object):
    def __init__(self, config, mode, device='cpu', **kwargs):
        config = edict(config)
        config.workers = config.workers // 2
        self.config = config
        nyu_conf = change_dataset(edict(config), 'nyu')
        kitti_conf = change_dataset(edict(config), 'kitti')

        # make nyu default for testing
        self.config = config = nyu_conf
        img_size = self.config.get("img_size", None)
        img_size = img_size if self.config.get(
            "do_input_resize", False) else None
        if mode == 'train':
            nyu_loader = DepthDataLoader(
                nyu_conf, mode, device=device, transform=preprocessing_transforms(mode, size=img_size)).data
            kitti_loader = DepthDataLoader(
                kitti_conf, mode, device=device, transform=preprocessing_transforms(mode, size=img_size)).data
            # It has been changed to repetitive roundrobin
            self.data = RepetitiveRoundRobinDataLoader(
                nyu_loader, kitti_loader)
        else:
            self.data = DepthDataLoader(nyu_conf, mode, device=device).data


def remove_leading_slash(s):
    if s[0] == '/' or s[0] == '\\':
        return s[1:]
    return s


class CachedReader:
    def __init__(self, shared_dict=None):
        if shared_dict:
            self._cache = shared_dict
        else:
            self._cache = {}

    def open(self, fpath):
        im = self._cache.get(fpath, None)
        if im is None:
            im = self._cache[fpath] = Image.open(fpath)
        return im


class ImReader:
    def __init__(self):
        pass

    # @cache
    def open(self, fpath):
        return Image.open(fpath)


class DataLoadPreprocess(Dataset):
    def __init__(self, config, mode, transform=None, is_for_online_eval=False, **kwargs):
        self.config = config
        if mode == 'online_eval':
            with open(config.filenames_file_eval, 'r') as f:
                self.filenames = f.readlines()
        else:
            with open(config.filenames_file, 'r') as f:
                self.filenames = f.readlines()

        self.filenames = self.filenames[:100] if config.get("debug", False) else self.filenames
        self.mode = mode
        self.transform = transform
        self.to_tensor = ToTensor(mode)
        self.is_for_online_eval = is_for_online_eval
        if config.use_shared_dict:
            self.reader = CachedReader(config.shared_dict)
        else:
            self.reader = ImReader()

    def postprocess(self, sample):
        return sample

    def __getitem__(self, idx):
        sample_path = self.filenames[idx]
        focal = float(sample_path.split()[2])
        sample = {}

        if self.mode == 'train':
            if self.config.dataset == 'kitti' and self.config.use_right and random.random() > 0.5:
                image_path = os.path.join(
                    self.config.data_path, remove_leading_slash(sample_path.split()[3]))
                depth_path = os.path.join(
                    self.config.gt_path, remove_leading_slash(sample_path.split()[4]))
            else:
                image_path = os.path.join(
                    self.config.data_path, remove_leading_slash(sample_path.split()[0]))
                depth_path = os.path.join(
                    self.config.gt_path, remove_leading_slash(sample_path.split()[1]))

            image = self.reader.open(image_path)
            depth_gt = self.reader.open(depth_path)
            w, h = image.size

            if self.config.do_kb_crop:
                height = image.height
                width = image.width
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                depth_gt = depth_gt.crop(
                    (left_margin, top_margin, left_margin + 1216, top_margin + 352))
                image = image.crop(
                    (left_margin, top_margin, left_margin + 1216, top_margin + 352))

            # Avoid blank boundaries due to pixel registration?
            # Train images have white border. Test images have black border.
            if self.config.dataset == 'nyu' and self.config.avoid_boundary:
                # print("Avoiding Blank Boundaries!")
                # We just crop and pad again with reflect padding to original size
                # original_size = image.size
                crop_params = get_white_border(np.array(image, dtype=np.uint8))
                image = image.crop((crop_params.left, crop_params.top, crop_params.right, crop_params.bottom))
                depth_gt = depth_gt.crop((crop_params.left, crop_params.top, crop_params.right, crop_params.bottom))

                # Use reflect padding to fill the blank
                image = np.array(image)
                image = np.pad(image, ((crop_params.top, h - crop_params.bottom), (crop_params.left, w - crop_params.right), (0, 0)), mode='reflect')
                image = Image.fromarray(image)

                depth_gt = np.array(depth_gt)
                depth_gt = np.pad(depth_gt, ((crop_params.top, h - crop_params.bottom), (crop_params.left, w - crop_params.right)), 'constant', constant_values=0)
                depth_gt = Image.fromarray(depth_gt)


            if self.config.do_random_rotate and (self.config.aug):
                random_angle = (random.random() - 0.5) * 2 * self.config.degree
                image = self.rotate_image(image, random_angle)
                depth_gt = self.rotate_image(
                    depth_gt, random_angle, flag=Image.NEAREST)

            image = np.asarray(image, dtype=np.float32) / 255.0
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=2)

            if self.config.dataset == 'nyu':
                depth_gt = depth_gt / 1000.0
            else:
                depth_gt = depth_gt / 256.0

            if self.config.aug and (self.config.random_crop):
                image, depth_gt = self.random_crop(
                    image, depth_gt, self.config.input_height, self.config.input_width)
            
            if self.config.aug and self.config.random_translate:
                # print("Random Translation!")
                image, depth_gt = self.random_translate(image, depth_gt, self.config.max_translation)

            image, depth_gt = self.train_preprocess(image, depth_gt)
            mask = np.logical_and(depth_gt > self.config.min_depth,
                                  depth_gt < self.config.max_depth).squeeze()[None, ...]
            sample = {'image': image, 'depth': depth_gt, 'focal': focal,
                      'mask': mask, **sample}

        else:
            if self.mode == 'online_eval':
                data_path = self.config.data_path_eval
            else:
                data_path = self.config.data_path

            image_path = os.path.join(
                data_path, remove_leading_slash(sample_path.split()[0]))
            image = np.asarray(self.reader.open(image_path),
                               dtype=np.float32) / 255.0

            if self.mode == 'online_eval':
                gt_path = self.config.gt_path_eval
                depth_path = os.path.join(
                    gt_path, remove_leading_slash(sample_path.split()[1]))
                has_valid_depth = False
                try:
                    depth_gt = self.reader.open(depth_path)
                    has_valid_depth = True
                except IOError:
                    depth_gt = False
                    # print('Missing gt for {}'.format(image_path))

                if has_valid_depth:
                    depth_gt = np.asarray(depth_gt, dtype=np.float32)
                    depth_gt = np.expand_dims(depth_gt, axis=2)
                    if self.config.dataset == 'nyu':
                        depth_gt = depth_gt / 1000.0
                    else:
                        depth_gt = depth_gt / 256.0

                    mask = np.logical_and(
                        depth_gt >= self.config.min_depth, depth_gt <= self.config.max_depth).squeeze()[None, ...]
                else:
                    mask = False

            if self.config.do_kb_crop:
                height = image.shape[0]
                width = image.shape[1]
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                image = image[top_margin:top_margin + 352,
                              left_margin:left_margin + 1216, :]
                if self.mode == 'online_eval' and has_valid_depth:
                    depth_gt = depth_gt[top_margin:top_margin +
                                        352, left_margin:left_margin + 1216, :]

            if self.mode == 'online_eval':
                sample = {'image': image, 'depth': depth_gt, 'focal': focal, 'has_valid_depth': has_valid_depth,
                          'image_path': sample_path.split()[0], 'depth_path': sample_path.split()[1],
                          'mask': mask}
            else:
                sample = {'image': image, 'focal': focal}

        if (self.mode == 'train') or ('has_valid_depth' in sample and sample['has_valid_depth']):
            mask = np.logical_and(depth_gt > self.config.min_depth,
                                  depth_gt < self.config.max_depth).squeeze()[None, ...]
            sample['mask'] = mask

        if self.transform:
            sample = self.transform(sample)

        sample = self.postprocess(sample)
        sample['dataset'] = self.config.dataset
        sample = {**sample, 'image_path': sample_path.split()[0], 'depth_path': sample_path.split()[1]}

        return sample

    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def random_crop(self, img, depth, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]

        return img, depth
    
    def random_translate(self, img, depth, max_t=20):
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        p = self.config.translate_prob
        do_translate = random.random()
        if do_translate > p:
            return img, depth
        x = random.randint(-max_t, max_t)
        y = random.randint(-max_t, max_t)
        M = np.float32([[1, 0, x], [0, 1, y]])
        # print(img.shape, depth.shape)
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        depth = cv2.warpAffine(depth, M, (depth.shape[1], depth.shape[0]))
        depth = depth.squeeze()[..., None]  # add channel dim back. Affine warp removes it
        # print("after", img.shape, depth.shape)
        return img, depth

    def train_preprocess(self, image, depth_gt):
        if self.config.aug:
            # Random flipping
            do_flip = random.random()
            if do_flip > 0.5:
                image = (image[:, ::-1, :]).copy()
                depth_gt = (depth_gt[:, ::-1, :]).copy()

            # Random gamma, brightness, color augmentation
            do_augment = random.random()
            if do_augment > 0.5:
                image = self.augment_image(image)

        return image, depth_gt

    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        if self.config.dataset == 'nyu':
            brightness = random.uniform(0.75, 1.25)
        else:
            brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug

    def __len__(self):
        return len(self.filenames)


class ToTensor(object):
    def __init__(self, 
                 mode, 
                 do_normalize=False, 
                 size=None,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):
        self.mode = mode
        self.normalize = transforms.Normalize(
            mean=mean, std=std) if do_normalize else nn.Identity()
        self.size = size
        if size is not None:
            self.resize = transforms.Resize(size=size)
        else:
            self.resize = nn.Identity()

    def __call__(self, sample):
        image, focal = sample['image'], sample['focal']
        image = self.to_tensor(image)
        image = self.normalize(image)
        image = self.resize(image)

        if self.mode == 'test':
            return {'image': image, 'focal': focal}

        depth = sample['depth']
        if self.mode == 'train':
            depth = self.to_tensor(depth)
            return {**sample, 'image': image, 'depth': depth, 'focal': focal}
        else:
            has_valid_depth = sample['has_valid_depth']
            image = self.resize(image)
            return {**sample, 'image': image, 'depth': depth, 'focal': focal, 'has_valid_depth': has_valid_depth}

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            # Ensure the transposed array is writable to avoid PyTorch warning
            np_img = pic.transpose((2, 0, 1)).copy()
            img = torch.from_numpy(np_img)
            return img

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img

class ALLODataLoadPreprocess(Dataset):
    def __init__(self, config, mode, transform=None, is_for_online_eval=False, **kwargs):
        self.config = config
        if mode == 'train':
            root = Path(config.data_path)
        elif mode == 'online_eval':
            root = Path(config.data_path_eval)

        columns = ['image_path', 'gt_depth_path']
        self.samples = DataFrame([{"image_path": str(frame),
                                "gt_depth_path": str(frame).replace('images', 'depth').replace('.png', '.exr')}
                        for frame in root.glob('**/images/*normal.png')],
                        columns=columns)
        self.samples = self.samples.sample(n=100, random_state=42) if config.get("debug", False) else self.samples

        self.mode = mode
        self.transform = transform
        self.to_tensor = ToTensor(mode)
        self.is_for_online_eval = is_for_online_eval
        if config.use_shared_dict:
            self.reader = CachedReader(config.shared_dict)
        else:
            self.reader = ImReader()

    def postprocess(self, sample):
        return sample

    def __getitem__(self, idx):
        image_path = self.samples.iloc[idx].image_path
        depth_path = self.samples.iloc[idx].gt_depth_path

        sample = {"image_path": image_path, "depth_path": depth_path, "has_valid_depth": True,
                  "focal": 1.0} #* focal is not used in ALLO dataset, so we set it to 1.0 as a placeholder
        image = self.reader.open(image_path)
        depth_gt = read_exr_depth_as_pil(depth_path)    #* read exr file
        seg_gt = self.reader.open(image_path.replace('images', 'segmentation_masks'))

        if self.mode == 'train':
            if self.config.do_random_rotate and (self.config.aug):
                random_angle = (random.random() - 0.5) * 2 * self.config.degree
                image = self.rotate_image(image, random_angle)
                depth_gt = self.rotate_image(
                    depth_gt, random_angle, flag=Image.NEAREST)
                seg_gt = self.rotate_image(seg_gt, random_angle, flag=Image.NEAREST)

            image = np.asarray(image, dtype=np.float32) / 255.0
            depth_gt = np.asarray(depth_gt, dtype=np.float32, copy=True)
            #* set depth to 0 if segmentation mask is 0 (background)
            seg_gt = np.asarray(seg_gt)
            depth_gt[seg_gt == 0] = 0
            depth_gt = np.expand_dims(depth_gt, axis=2)

            if self.config.aug and (self.config.random_crop):
                image, depth_gt = self.random_crop(
                    image, depth_gt, self.config.input_height, self.config.input_width)

            image, depth_gt = self.train_preprocess(image, depth_gt)

        else:
            if self.mode == 'online_eval':
                image = np.asarray(image, dtype=np.float32) / 255.0
                depth_gt = np.asarray(depth_gt, dtype=np.float32, copy=True)
                #* set depth to 100 if segmentation mask is 0 (background)
                seg_gt = np.asarray(seg_gt)
                depth_gt[seg_gt == 0] = 100
                depth_gt = np.expand_dims(depth_gt, axis=2)

        mask = np.logical_and(depth_gt > self.config.min_depth,
                                depth_gt < self.config.max_depth).squeeze()[None, ...]
        sample = {'image': image, 'depth': depth_gt, 'mask': mask, **sample}
        if mask.sum() == 0:
            sample['has_valid_depth'] = False

        if self.transform:
            sample = self.transform(sample)

        sample = self.postprocess(sample)   #* doesn't do anything
        sample['dataset'] = self.config.dataset
        return sample

    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def random_crop(self, img, depth, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]

        return img, depth
    
    def random_translate(self, img, depth, max_t=20):
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        p = self.config.translate_prob
        do_translate = random.random()
        if do_translate > p:
            return img, depth
        x = random.randint(-max_t, max_t)
        y = random.randint(-max_t, max_t)
        M = np.float32([[1, 0, x], [0, 1, y]])
        # print(img.shape, depth.shape)
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        depth = cv2.warpAffine(depth, M, (depth.shape[1], depth.shape[0]))
        depth = depth.squeeze()[..., None]  # add channel dim back. Affine warp removes it
        # print("after", img.shape, depth.shape)
        return img, depth

    def train_preprocess(self, image, depth_gt):
        if self.config.aug:
            # Random flipping
            do_flip = random.random()
            if do_flip > 0.5:
                image = (image[:, ::-1, :]).copy()
                depth_gt = (depth_gt[:, ::-1, :]).copy()

            # Random gamma, brightness, color augmentation
            do_augment = random.random()
            if do_augment > 0.5:
                image = self.augment_image(image)

        return image, depth_gt

    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        if self.config.dataset == 'nyu':
            brightness = random.uniform(0.75, 1.25)
        else:
            brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug

    def __len__(self):
        return len(self.samples)


class STUMixDataLoadPreprocess(Dataset):
    def __init__(self, config, mode, transform=None, is_for_online_eval=False, **kwargs):
        self.config = config
        columns = ['image_path', 'lidar_path', 'dataset', 'sequence']
        
        semantickitti_sequences = ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"]
        SSCBenchKitti360_sequences = ["2013_05_28_drive_0000_sync", "2013_05_28_drive_0002_sync", "2013_05_28_drive_0003_sync",
                                        "2013_05_28_drive_0004_sync", "2013_05_28_drive_0005_sync", "2013_05_28_drive_0007_sync",
                                        "2013_05_28_drive_0010_sync"]
        STU_sequences = ["201", "206"]
        
        if mode == 'train':
            semKittiRoot = Path(config.semantickitti_data_path) / "sequences"
            SSCBenchKitti360Root = Path(config.SSCBenchKitti360_data_path)
            STURoot = Path(config.STU_data_path) / "train"
            
            #* Semantic KITTI
            self.semKittiSamples = DataFrame([{"image_path": str(frame),
                                "lidar_path": (str(frame)
                                               .replace('image_2', 'velodyne')
                                               .replace('.png', '.bin')),
                                "dataset": "semantickitti",
                                "sequence": (str(frame)).split("/")[-3]}
                        for frame in semKittiRoot.glob('**/image_2/*.png')],
                        columns=columns)
            self.semKittiSamples = self.semKittiSamples[self.semKittiSamples["sequence"].isin(semantickitti_sequences)]
            self.semKittiSamples = self.semKittiSamples[::2]
            #* SSCBenchKitti360
            self.SSCBenchKitti360Samples = DataFrame([{"image_path": str(frame),
                                "lidar_path": (str(frame)
                                               .replace('data_2d_raw', 'data_3d_raw')
                                               .replace('image_00/data_rect', 'velodyne_points/data')
                                               .replace('.png', '.bin')),
                                "dataset": "SSCBenchKitti360",
                                "sequence": (str(frame)).split("/")[-4]}
                        for frame in SSCBenchKitti360Root.glob('data_2d_raw/**/image_00/data_rect/*.png')],
                        columns=columns)
            self.SSCBenchKitti360Samples = self.SSCBenchKitti360Samples[self.SSCBenchKitti360Samples["sequence"].isin(SSCBenchKitti360_sequences)]
            self.SSCBenchKitti360Samples = self.SSCBenchKitti360Samples[::2]
            #* STU
            self.STUSamples = DataFrame([{"image_path": str(frame),
                                "lidar_path": (str(frame)
                                               .replace('port_a_cam_0', 'velodyne')
                                               .replace('.png', '.bin')),
                                "dataset": "STU",
                                "sequence": (str(frame)).split("/")[-3]}
                        for frame in STURoot.glob('**/port_a_cam_0/*.png')],
                        columns=columns)
            self.STUSamples = self.STUSamples[self.STUSamples["sequence"].isin(STU_sequences)]
            
            #? Combine all samples
            self.samples = pd.concat([self.semKittiSamples, self.SSCBenchKitti360Samples, self.STUSamples])
        elif mode == 'online_eval':
            root = Path(config.STU_data_path) / "val"
            
            #* STU only
            self.samples = DataFrame([{"image_path": str(frame),
                                "lidar_path": (str(frame)
                                               .replace('port_a_cam_0', 'velodyne')
                                               .replace('.png', '.bin')),
                                "dataset": "STU"}
                        for frame in root.glob('**/port_a_cam_0/*.png')],
                        columns=columns)

        self.samples = self.samples.sample(n=100, random_state=42) if config.get("debug", False) else self.samples

        self.mode = mode
        self.transform = transform
        self.to_tensor = ToTensor(mode)
        self.is_for_online_eval = is_for_online_eval
        if config.use_shared_dict:
            self.reader = CachedReader(config.shared_dict)
        else:
            self.reader = ImReader()

    def postprocess(self, sample):
        return sample

    def __len__(self):
        return len(self.samples)

    def get_sequence_id(self, image_path, stop_name):
        names = image_path.split("/")
        sequence_path = []
        for name in names:
            if name == stop_name:
                break
            sequence_path.append(name)
        sequence_path = "/".join(sequence_path)
        return sequence_path


    @staticmethod
    def get_calib_semantickitti(calib_path):
        """calib.txt: Calibration data for the cameras: P0/P1 are the 3x4 projection
            matrices after rectification. Here P0 denotes the left and P1 denotes the
            right camera. Tr transforms a point from velodyne coordinates into the
            left rectified camera coordinate system. In order to map a point X from the
            velodyne scanner to a point x in the i'th image plane, you thus have to
            transform it like:
            x = Pi * Tr * X
            - 'image_00': left rectified grayscale image sequence
            - 'image_01': right rectified grayscale image sequence
            - 'image_02': left rectified color image sequence
            - 'image_03': right rectified color image sequence
        """
        calib_all = {}
        with open(calib_path, "r") as f:
            for line in f.readlines():
                if line == "\n":
                    break
                key, value = line.split(":", 1)
                calib_all[key] = np.array([float(x) for x in value.split()])

        # reshape matrices
        calib_out = {}
        calib_out["P2"] = np.identity(4)  # 4x4 matrix
        calib_out["P3"] = np.identity(4)  # 4x4 matrix
        calib_out["P2"][:3, :4] = calib_all["P2"].reshape(3, 4)
        calib_out["P3"][:3, :4] = calib_all["P3"].reshape(3, 4)
        calib_out["Tr"] = np.identity(4)  # 4x4 matrix
        calib_out["Tr"][:3, :4] = calib_all["Tr"].reshape(3, 4) 
        
        return calib_out


    @staticmethod
    def get_calib_SSCBenchKitti360():
        P2 = torch.tensor([
            [552.554261, 0.000000, 682.049453, 0.000000],
            [0.000000, 552.554261, 238.769549, 0.000000],
            [0.000000, 0.000000, 1.000000, 0.000000]
        ]).reshape(3, 4)

        cam2velo = torch.tensor([   
            [0.04307104361, -0.08829286498, 0.995162929, 0.8043914418],
            [-0.999004371, 0.007784614041, 0.04392796942, 0.2993489574],
            [-0.01162548558, -0.9960641394, -0.08786966659, -0.1770225824],
            [0, 0, 0, 1]
        ]).reshape(4, 4)

        velo2cam = cam2velo.inverse()
        calib_out = {}
        calib_out["P2"] = torch.eye(4)  # 4x4 matrix
        calib_out["P2"][:3, :4] = P2.reshape(3, 4)
        calib_out["Tr"] = torch.eye(4)
        calib_out["Tr"][:3, :4] = velo2cam[:3, :4]
        
        return calib_out

    @staticmethod
    def get_calib_STU():
        calib_out = {}
        calib_out["P2"] = torch.eye(4)  # 4x4 matrix
        #* P matrix for both intrinsic and distortion (K, D)
        calib_out["P2"][:3, :4] = torch.tensor(
            [[1678.93384, 0.0, 918.44184, 0.0],
            [0.0, 1779.26013, 644.71356, 0.0],
            [0.0, 0.0, 1.0, 0.0]]
        )
        calib_out["Tr"] = torch.eye(4)  # 4x4 matrix
        #* LiDAR -> Camera
        calib_out["Tr"][:3, :4] = torch.tensor(
            [[0.014493257457, -0.999718233867, -0.018798892575, -0.006080995796],
            [0.010622477153, 0.018953749566, -0.999763931314, -0.400777062539],
            [0.999838541199, 0.014290145246, 0.010894185671, -0.761577584775]]
        )
        
        return calib_out


    @staticmethod
    def project_points(points, rots, trans, intrins):
        # from lidar to camera
        points = points.view(-1, 1, 3) # N, 1, 3
        points = points - trans.view(1, -1, 3) # N, b, 3
        inv_rots = rots.inverse().unsqueeze(0) # 1, b, 3, 3
        points = (inv_rots @ points.unsqueeze(-1)) # N, b, 3, 1
        # the intrinsic matrix is [4, 4] for kitti and [3, 3] for nuscenes 
        if intrins.shape[-1] == 4:
            points = torch.cat((points, torch.ones((points.shape[0], points.shape[1], 1, 1))), dim=2) # N, b, 4, 1
            points = (intrins.unsqueeze(0) @ points).squeeze(-1) # N, b, 4
        else:
            points = (intrins.unsqueeze(0) @ points).squeeze(-1)

        points_d = points[..., 2:3] # N, b, 1
        points_uv = points[..., :2] / points_d # N, b, 2
        points_uvd = torch.cat((points_uv, points_d), dim=2)
        
        return points_uvd


    def get_depth_from_lidar(self, image, lidar_points, intrins, rots, trans):
        projected_points = self.project_points(lidar_points, rots, trans, intrins)
        
        img_h, img_w = image.height, image.width
        valid_mask = (projected_points[..., 0] >= 0) & \
                    (projected_points[..., 1] >= 0) & \
                    (projected_points[..., 0] <= img_w - 1) & \
                    (projected_points[..., 1] <= img_h - 1) & \
                    (projected_points[..., 2] > 0)
        
        #? Get depth from LiDAR points
        gt_depth = torch.zeros((img_h, img_w))
        projected_points = projected_points[:, 0]
        valid_mask = valid_mask[:, 0]
        valid_points = projected_points[valid_mask]
        # sort
        depth_order = torch.argsort(valid_points[:, 2], descending=True)
        valid_points = valid_points[depth_order]
        # fill in
        gt_depth[valid_points[:, 1].round().long(), 
                valid_points[:, 0].round().long()] = valid_points[:, 2].float()
        
        #? Convert back to PIL image
        gt_depth = gt_depth.numpy()
        depth_gt = Image.fromarray(gt_depth)
        
        return depth_gt


    def __getitem__(self, idx):
        image_path = self.samples.iloc[idx].image_path
        lidar_path = self.samples.iloc[idx].lidar_path
        dataset = self.samples.iloc[idx].dataset
        
        sample = {"image_path": image_path, "has_valid_depth": True, "focal": 1.0} #* focal is not used in ALLO dataset, so we set it to 1.0 as a placeholder
        image = self.reader.open(image_path)
        lidar_points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
        lidar_points = torch.from_numpy(lidar_points[:, :3]).float()
        
        #? Get sequence id
        if dataset == "semantickitti":
            #* sequence path is before "image_2"
            sequence_path = self.get_sequence_id(image_path, "image_2")
        elif dataset == "SSCBenchKitti360":
            sequence_path = self.get_sequence_id(image_path, "image_00")
        elif dataset == "STU":
            sequence_path = self.get_sequence_id(image_path, "port_a_cam_0")
        
        #? Get calibration info
        if dataset in ["semantickitti", "SSCBenchKitti360"]:
            if dataset == "semantickitti":
                calib_path = Path(sequence_path) / "calib.txt"
                calib = self.get_calib_semantickitti(calib_path)
                intrins = torch.from_numpy(calib["P2"]).unsqueeze(0)
                lidar2cam = torch.from_numpy(calib["Tr"]).unsqueeze(0)
            
            elif dataset == "SSCBenchKitti360":
                calib = self.get_calib_SSCBenchKitti360()
                intrins = calib["P2"].unsqueeze(0)
                lidar2cam = calib["Tr"].unsqueeze(0)

            cam2lidar = lidar2cam.inverse()
            rots = cam2lidar[:, :3, :3]
            trans = cam2lidar[:, :3, 3]
            
            depth_gt = self.get_depth_from_lidar(image, lidar_points, intrins, rots, trans)

            if dataset == "semantickitti" and self.config.do_kb_crop:
                height = image.height
                width = image.width
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                depth_gt = depth_gt.crop(
                    (left_margin, top_margin, left_margin + 1216, top_margin + 352))
                image = image.crop(
                    (left_margin, top_margin, left_margin + 1216, top_margin + 352))
        
        elif dataset == "STU":
            calib = self.get_calib_STU()
            intrins = calib["P2"].unsqueeze(0)
            lidar2cam = calib["Tr"].unsqueeze(0)
            cam2lidar = lidar2cam.inverse()
            rots = cam2lidar[:, :3, :3]
            trans = cam2lidar[:, :3, 3]
            
            depth_gt = self.get_depth_from_lidar(image, lidar_points, intrins, rots, trans)
        
        else:
            raise ValueError(f"Dataset {dataset} not supported")

        if self.mode == 'train':
            if self.config.do_random_rotate and (self.config.aug):
                random_angle = (random.random() - 0.5) * 2 * self.config.degree
                image = self.rotate_image(image, random_angle)
                depth_gt = self.rotate_image(
                    depth_gt, random_angle, flag=Image.NEAREST)

            # Resize with aspect ratio and pad on bottom-right to target size
            image, depth_gt = resize_and_pad_pil_pair(
                image, depth_gt, self.config.input_height, self.config.input_width
            )

            image = np.asarray(image, dtype=np.float32) / 255.0
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=2)
            
            image, depth_gt = self.train_preprocess(image, depth_gt)

        else:
            if self.mode == 'online_eval':
                # Resize with aspect ratio and pad on bottom-right to target size
                image, depth_gt = resize_and_pad_pil_pair(
                    image, depth_gt, self.config.input_height, self.config.input_width
                )
                image = np.asarray(image, dtype=np.float32) / 255.0
                depth_gt = np.asarray(depth_gt, dtype=np.float32)
                depth_gt = np.expand_dims(depth_gt, axis=2) #* (H, W, 1)

        mask = np.logical_and(depth_gt > self.config.min_depth,
                                depth_gt < self.config.max_depth).squeeze()[None, ...]
        sample = {'image': image, 'depth': depth_gt, 'mask': mask, **sample}
        if mask.sum() == 0:
            sample['has_valid_depth'] = False

        if self.transform:
            sample = self.transform(sample)

        sample = self.postprocess(sample)   #* doesn't do anything
        sample['dataset'] = self.config.dataset
        return sample

    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def resize_pad_image(self, image, height, width):
        image = image.resize((width, height), Image.BICUBIC)
        
        return image

    def random_crop(self, img, depth, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]

        return img, depth
    
    def random_translate(self, img, depth, max_t=20):
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        p = self.config.translate_prob
        do_translate = random.random()
        if do_translate > p:
            return img, depth
        x = random.randint(-max_t, max_t)
        y = random.randint(-max_t, max_t)
        M = np.float32([[1, 0, x], [0, 1, y]])
        # print(img.shape, depth.shape)
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        depth = cv2.warpAffine(depth, M, (depth.shape[1], depth.shape[0]))
        depth = depth.squeeze()[..., None]  # add channel dim back. Affine warp removes it
        # print("after", img.shape, depth.shape)
        return img, depth

    def train_preprocess(self, image, depth_gt):
        if self.config.aug:
            # Random flipping
            do_flip = random.random()
            if do_flip > 0.5:
                image = (image[:, ::-1, :]).copy()
                depth_gt = (depth_gt[:, ::-1, :]).copy()

            # Random gamma, brightness, color augmentation
            do_augment = random.random()
            if do_augment > 0.5:
                image = self.augment_image(image)

        return image, depth_gt

    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        if self.config.dataset == 'nyu':
            brightness = random.uniform(0.75, 1.25)
        else:
            brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug