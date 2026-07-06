from data.dataset import SOPTripletDataset

dataset = SOPTripletDataset(
    dataset_root="dataset/Stanford_Online_Products",
    annotation_file="dataset/Stanford_Online_Products/Ebay_train.txt"
)

print("\nTotal Samples :", len(dataset))
print("Total Classes :", len(dataset.classes))

sizes = [len(dataset.class_to_images[c]) for c in dataset.classes]

print("Minimum Images/Class :", min(sizes))
print("Maximum Images/Class :", max(sizes))
print("Average Images/Class :", sum(sizes)/len(sizes))