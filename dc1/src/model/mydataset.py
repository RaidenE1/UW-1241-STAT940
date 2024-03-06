import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np

train_mean, train_std = (0.43916297, 0.42551237, 0.40391925), (0.26868406, 0.27018538, 0.2759704)
test_mean, test_std = (0.43916297, 0.42551237, 0.40391925), (0.26868406, 0.27018538, 0.2759704)

class TrainDataset(Dataset):
    def __init__(self, image_dir, images, transform=None, labels=None):
        self.image_dir = image_dir
        self.images = images
        self.labels = labels
        self.transform = transform
        # self.normal_transform = transforms.Compose(
        #     [
        #         transforms.RandomCrop(32, padding=4),
        #         transforms.ToTensor(),
        #         transforms.Normalize(train_mean, train_std),
        #     ]
        # )
        # self.horizontal = transforms.Compose(
        #     [
        #         transforms.RandomCrop(32, padding=4),
        #         transforms.RandomHorizontalFlip(1),
        #         transforms.ToTensor(),
        #         transforms.Normalize(train_mean, train_std),
        #     ]
        # )
        # self.vertical = transforms.Compose(
        #     [
        #         transforms.RandomCrop(32, padding=4),
        #         transforms.RandomVerticalFlip(1),
        #         transforms.ToTensor(),
        #         transforms.Normalize(train_mean, train_std),
        #     ]
        # )
        # self.rotation = transforms.Compose(
        #     [
        #         transforms.RandomCrop(32, padding=4),
        #         transforms.RandomRotation(30),
        #         transforms.ToTensor(),
        #         transforms.Normalize(train_mean, train_std),
        #     ]
        # )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_name).convert('RGB')
        # strategy = idx // len(self.images)
        # if strategy == 0:
        #     image = self.normal_transform(image)
        # elif strategy == 1:
        #     image = self.horizontal(image)
        # elif strategy == 2:
        #     image = self.vertical(image)
        # else:
        #     image = self.rotation(image)
        if self.transform:
            image = self.transform(image)
        image_name = self.images[idx]
        image_label = self.labels.get(int(image_name.split('.')[0]), None)

        return image, image_label
    
    
class ValDataset(Dataset):
    def __init__(self, image_dir, images, transform=None, labels=None):
        self.image_dir = image_dir
        self.images = images
        self.transform = transform
        self.labels = labels
        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)
        image_name = self.images[idx]
        image_label = self.labels.get(int(image_name.split('.')[0]), None)

        return image, image_label


class TestDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        image_name = self.images[idx]
        image_prefix = image_name.split('.')[0]

        return image, image_prefix
    
# def random_hsv(image):
#     random_h = np.random.uniform(*c.hue_delta)
#     random_s = np.random.uniform(*c.saturation_scale)
#     random_v = np.random.uniform(*c.brightness_scale)

#     image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     image_hsv[:, :, 0] = image_hsv[:, :, 0] + random_h % 360.0  # hue
#     image_hsv[:, :, 1] = np.minimum(image_hsv[:, :, 1] * random_s, 1.0)  # saturation
#     image_hsv[:, :, 2] = np.minimum(image_hsv[:, :, 2] * random_v, 255.0)  # brightness

#     return cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)