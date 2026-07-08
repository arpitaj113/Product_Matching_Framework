import os
import sys
import pickle
import faiss

import torch
from PIL import Image


# --------------------------------------------------
# Add project root to Python path
# --------------------------------------------------

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)


from models.embedding_network import EmbeddingNet
from data.transforms import test_transform


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)


MODEL_PATH = "/content/drive/MyDrive/ProductMatchingModels/triplet_model.pth"


INDEX_PATH = os.path.join(
    BASE_DIR,
    "embeddings",
    "faiss.index"
)


IMAGE_PATHS = os.path.join(
    BASE_DIR,
    "embeddings",
    "image_paths.pkl"
)


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# --------------------------------------------------
# Load Model
# --------------------------------------------------

print("Loading model...")


model = EmbeddingNet(
    embedding_dim=128,
    pretrained=False
).to(DEVICE)


model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)


model.eval()


print("Model Loaded!")


# --------------------------------------------------
# Load FAISS Index
# --------------------------------------------------

print("Loading FAISS index...")


index = faiss.read_index(
    INDEX_PATH
)


print("FAISS Index Loaded!")


# --------------------------------------------------
# Load Image Paths
# --------------------------------------------------

print("Loading image paths...")


with open(IMAGE_PATHS, "rb") as f:
    image_paths = pickle.load(f)


print("Everything Loaded Successfully!")


# --------------------------------------------------
# Generate Image Embedding
# --------------------------------------------------

def get_embedding(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")


    image = test_transform(
        image
    )


    image = image.unsqueeze(0).to(
        DEVICE
    )


    with torch.no_grad():

        embedding = model(
            image
        )


    embedding = embedding.cpu().numpy().astype(
        "float32"
    )


    return embedding



# --------------------------------------------------
# Search Similar Images
# --------------------------------------------------

def search_image(query_image, top_k=5):


    embedding = get_embedding(
        query_image
    )


    distances, indices = index.search(
        embedding,
        top_k
    )


    results = []


    for idx, dist in zip(
        indices[0],
        distances[0]
    ):


        if idx == -1:
            continue


        results.append(
            {
                "image_path": image_paths[idx],
                "distance": float(dist)
            }
        )


    return results




# --------------------------------------------------
# Test
# --------------------------------------------------

if __name__ == "__main__":


    query = input(
        "Enter image path: "
    )


    results = search_image(
        query
    )


    print("\nTop Matches\n")


    for i, result in enumerate(results):

        print(
            i + 1,
            result["image_path"],
            result["distance"]
        )