import torch

from models.embedding_network import EmbeddingNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using:", device)

model = EmbeddingNet()

model = model.to(device)

dummy = torch.randn(4,3,224,224).to(device)

embedding = model(dummy)

print()

print("Embedding Shape:", embedding.shape)

print()

print("Norm of first embedding:")

print(torch.norm(embedding[0]))