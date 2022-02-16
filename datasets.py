import random
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class Datasets(Dataset):
    def __init__(self, paths_file, image_size):
        self.image_size = image_size

        self.input_image_paths = []
        self.mask_image_paths = []
        self.labels = []
        
        with open(paths_file, 'r') as f:
            lines = f.readlines()
            for l in lines:
                parts = l.rstrip().split(' ')
                self.input_image_paths.append(parts[0])
                self.mask_image_paths.append(parts[1])
                self.labels.append(int(parts[2]))

        # ----------
        #  TODO: Transforms for data augmentation (more augmentations should be added)
        # ----------
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor()])

    def __getitem__(self, item):
        # ----------
        # Read input image
        # ----------
        input_file_name = self.input_image_paths[item]
        input = cv2.imread(input_file_name)

        height, width, _ = input.shape

        # ----------
        # Read mask
        # ----------
        mask_file_name = self.mask_image_paths[item]
        if (mask_file_name == "None"):
            mask= np.zeros((height, width), np.uint8) # a totally black mask for real image
        else:
            mask = cv2.imread(mask_file_name, cv2.IMREAD_GRAYSCALE)

        # ----------
        # Apply transform (the same for both image and mask)
        # ----------
        seed = np.random.randint(2147483647) # make a seed with numpy generator 

        random.seed(seed)
        torch.manual_seed(seed)
        input = self.transform(torch.tensor(input).permute(2, 0, 1)) # permute is a must, as pytorch accepts [channel, height, width] format instead of [height, width, channel]

        random.seed(seed)
        torch.manual_seed(seed)
        mask = self.transform(torch.tensor(mask))

        ret = {'input': input, 'mask': mask, 'label': self.labels[item]}

        return ret

    def __len__(self):
        return len(self.input_image_paths)
