import os
import pickle

# ----------------------------
# Files
# ----------------------------

PICKLE_PATH = "embeddings/merged/image_paths.pkl"

# This is the Windows prefix we want to remove
OLD_PREFIX = r"D:\C-DAC_projects\Product_Matching_Framework\\"

# ----------------------------
# Load pickle
# ----------------------------

with open(PICKLE_PATH, "rb") as f:
    image_paths = pickle.load(f)

print(f"Loaded {len(image_paths)} image paths.")

# ----------------------------
# Convert paths
# ----------------------------

new_paths = []

for path in image_paths:

    # Convert to relative path
    relative_path = os.path.relpath(
        path,
        start=OLD_PREFIX.rstrip("\\")
    )

    # Use forward slashes (Linux compatible)
    relative_path = relative_path.replace("\\", "/")

    new_paths.append(relative_path)

# ----------------------------
# Backup old pickle
# ----------------------------

os.rename(
    PICKLE_PATH,
    "embeddings/merged/image_paths_backup.pkl"
)

# ----------------------------
# Save new pickle
# ----------------------------

with open(PICKLE_PATH, "wb") as f:
    pickle.dump(new_paths, f)

print("\nDone!")
print("Backup created:")
print("embeddings/merged/image_paths_backup.pkl")
