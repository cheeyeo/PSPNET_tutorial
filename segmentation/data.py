from dataclasses import dataclass
from collections import namedtuple
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt


# Represent single label of dataset
@dataclass
class Label:
    name: str
    train_id: int
    color: tuple


def drivables():
    drivables = [ 
        Label("direct", 0, (219, 94, 86)),        # red
        Label("alternative", 1, (86, 211, 219)),  # cyan
        Label("background", 2, (0, 0, 0)),        # black          
    ]

    return drivables


def train_id_to_colour():
    """
    Returns list of colours based on training instance id
    
    Only select train id between -1 and 255
    """
    train_id_to_color = [c.color for c in drivables() if (c.train_id != -1 and c.train_id != 255)]

    return np.array(train_id_to_color)


class BDD100K_dataset(Dataset):
    def __init__(self, images, labels, tf=None):
        super(BDD100K_dataset, self).__init__()
        self.images = images
        self.labels = labels
        self.tf = tf
    

    def __len__(self):
        return self.images.shape[0]
    

    def __getitem__(self, index):
        rgb_image = self.images[index]
        if self.tf is not None:
            rgb_image = self.tf(rgb_image)
        
        label_image = torch.from_numpy(self.labels[index]).long()
        return rgb_image, label_image


def get_datasets(images, labels):
    data = BDD100K_dataset(images, labels, tf=preprocess_transform())
    total_count = len(data)
    train_count = int(0.7 * total_count)
    valid_count = int(0.2 * total_count)
    test_count = total_count - train_count - valid_count
    train_set, val_set, test_set = torch.utils.data.random_split(data, (train_count, valid_count, test_count), generator=torch.Generator().manual_seed(1))
    return train_set, val_set, test_set
    

def get_dataloaders(train_set, val_set, test_set):
    """DON'T SHUFFLE THE VAL AND TEST SETS"""
    train_dataloader = DataLoader(train_set, batch_size=8, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_set, batch_size=8)
    test_dataloader = DataLoader(test_set, batch_size=8)
    return train_dataloader, val_dataloader, test_dataloader


def preview_data(images, labels, idx):
    f, (ax0, ax1) = plt.subplots(1, 2, figsize=(20, 40))
    ax0.imshow(images[idx])
    ax0.set_title("Image")
    ax1.imshow(labels[idx])
    ax1.set_title("Label")
    plt.show()


def preview_dataset(input_dataset):
    train_id_to_color = train_id_to_colour()
    rgb_image, label = input_dataset[int(np.random.uniform(len(input_dataset)))]
    rgb_image = inverse_transform()(rgb_image).permute(1, 2, 0).cpu().detach().numpy()
    label = label.cpu().detach().numpy()
    print(label)

    # plot sample image
    f, (ax0, ax1) = plt.subplots(1, 2, figsize=(20, 10))
    ax0.imshow(rgb_image)
    ax0.set_title("Image")
    ax0.axis("off")
    ax1.imshow(train_id_to_color[label])
    ax1.set_title("Label")
    ax1.axis("off")
    plt.show()


def preprocess_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.56, 0.406), std=(0.229, 0.224, 0.225))
    ])


def inverse_transform():
    return transforms.Compose([
        transforms.Normalize((-0.485/0.229, -0.456/0.224, -0.406/0.225), (1/0.229, 1/0.224, 1/0.225))
    ])


if __name__ == "__main__":
    images = np.load("dataset/image_180_320.npy")
    print(len(images))
    labels = np.load("dataset/label_180_320.npy")
    preview_data(images, labels, 202)

