import matplotlib.pyplot as plt
import torch

from data.dataset import SOPTripletDataset
from data.transforms import train_transform

DATASET_ROOT = "dataset/Stanford_Online_Products"

TRAIN_FILE = (
    "dataset/Stanford_Online_Products/Ebay_train.txt"
)

dataset = SOPTripletDataset(
    dataset_root=DATASET_ROOT,
    annotation_file=TRAIN_FILE,
    transform=train_transform
)


def denormalize(img):

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229,0.224,0.225]).view(3,1,1)

    img = img * std + mean

    img = img.clamp(0,1)

    return img


import random

idx = random.randint(0, len(dataset)-1)

anchor, positive, negative, a_cls, n_cls = dataset[idx]

anchor = denormalize(anchor)
positive = denormalize(positive)
negative = denormalize(negative)

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(anchor.permute(1,2,0))
plt.title(f"Anchor\nClass {a_cls}")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(positive.permute(1,2,0))
plt.title(f"Positive\nClass {a_cls}")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(negative.permute(1,2,0))
plt.title(f"Negative\nClass {n_cls}")
plt.axis("off")

plt.show()