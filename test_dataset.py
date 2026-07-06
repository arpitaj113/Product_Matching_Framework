from torch.utils.data import DataLoader

from data.dataset import SOPTripletDataset
from data.transforms import train_transform

DATASET_ROOT = "dataset/Stanford_Online_Products"

TRAIN_FILE = (
    "dataset/Stanford_Online_Products/"
    "Ebay_train.txt"
)

dataset = SOPTripletDataset(
    dataset_root=DATASET_ROOT,
    annotation_file=TRAIN_FILE,
    transform=train_transform
)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)

anchor, positive, negative, ac, nc = next(iter(loader))

print()

print(anchor.shape)
print(positive.shape)
print(negative.shape)

print()

print(ac)
print(nc)