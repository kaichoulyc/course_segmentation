import numpy as np
import cv2
import os
import torch
from torch.utils.data import DataLoader, Dataset
import glob
import matplotlib.pyplot as plt
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose, Normalize
)


def light_aug(p = 0.7):
    return Compose([
        RandomRotate90(),
        Flip(),
        Transpose(),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=0.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=0.1),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomContrast(),
            RandomBrightness(),
        ], p=0.3),
        HueSaturationValue(p=0.3),
        ], p = p)
def strong(p=1):
    return Compose([
        Normalize()], p=p)

def processing_mask(mask):
    mask = cv2.resize(mask, (256, 256))
    mask = np.expand_dims(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY), axis = 2)
    mask = np.transpose(mask, (2,0,1))
    mask = torch.from_numpy(mask)
    return mask

def pre_processing(img, mask, side_size, mode = 'train'):
    img = cv2.resize(img, (side_size, side_size))
    mask = cv2.resize(mask, (side_size, side_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if mode == 'train':
        augs = light_aug()
        augmented = augs(image = img, mask = mask)
        img = augmented["image"]
        mask = augmented["mask"]
    mask = np.expand_dims(mask, axis = 2)
    mask = mask.astype(np.bool)
#     norm = strong()
#     img = norm(image = img)
#     img = img["image"]
    img = np.transpose(img, (2,0,1))
    mask = np.transpose(mask, (2,0,1))
    img = torch.from_numpy(img).float()
    mask = torch.from_numpy(mask.astype(np.uint8)).float()
#     mask = torch.from_numpy(mask).float()
    return img, mask

def claim_generator(path, batch_size, workers, side_size = 256, mode = 'train'):
    
    image_names = os.listdir(os.path.join(path, 'images'))   
    
    for i in range(len(image_names)):
        image_names[i] = image_names[i].split('.')[0]
    
    some_set = PeopleDataset(path, image_names, side_size, mode)
    some_loader = DataLoader(some_set, batch_size=batch_size, num_workers=workers)
    
    return some_loader

    

class PeopleDataset(Dataset):
    
    def __init__(self, path, images, side_size, mode = 'train'):
        
        self.path = path
        self.len = len(images)
        self.imgs = images
        self.mode = mode
        self.side_size = side_size

    def __len__(self):
        return self.len
    
        
    def __getitem__(self, index):
        
        
        x = cv2.imread(next(glob.iglob(os.path.join(self.path + '/images', self.imgs[index]) + '.*')))
        y = cv2.imread(next(glob.iglob(os.path.join(self.path + '/masks', self.imgs[index]) + '.*')), cv2.IMREAD_GRAYSCALE)
        x, y = pre_processing(x, y, self.side_size, self.mode)
        
        
        
        return x, y
