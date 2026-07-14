import pickle

with open("embeddings/merged/image_paths.pkl", "rb") as f:
    paths = pickle.load(f)

print("Total image paths:", len(paths))

print("\nFirst 5 paths:\n")

for i in range(5):
    print(paths[i])