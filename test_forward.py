from torch.utils.data import DataLoader

from data.dataset import SOPTripletDataset
from data.transforms import train_transform
from models.embedding_network import EmbeddingNet

dataset = SOPTripletDataset(
    dataset_root="dataset/Stanford_Online_Products",
    annotation_file="dataset/Stanford_Online_Products/Ebay_train.txt",
    transform=train_transform
)

loader = DataLoader(dataset, batch_size=4, shuffle=True)

anchor, positive, negative, _, _ = next(iter(loader))

model = EmbeddingNet()

anchor_embedding = model(anchor)
positive_embedding = model(positive)
negative_embedding = model(negative)

print("Anchor:", anchor_embedding.shape)
print("Positive:", positive_embedding.shape)
print("Negative:", negative_embedding.shape)