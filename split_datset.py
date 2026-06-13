import os, shutil, random

# Paths
dataset_dir = 'dataset'
train_dir = os.path.join(dataset_dir, 'train')
test_dir = os.path.join(dataset_dir, 'test')

# Categories
categories = ['Organic', 'Plastic', 'Metal', 'E-waste']

# Create train/test folders
for cat in categories:
    os.makedirs(os.path.join(train_dir, cat), exist_ok=True)
    os.makedirs(os.path.join(test_dir, cat), exist_ok=True)

# Split ratio (80% train, 20% test)
split_ratio = 0.8

for cat in categories:
    cat_dir = os.path.join(dataset_dir, cat)
    images = os.listdir(cat_dir)
    random.shuffle(images)
    split_point = int(len(images) * split_ratio)

    train_images = images[:split_point]
    test_images = images[split_point:]

    for img in train_images:
        shutil.copy(os.path.join(cat_dir, img), os.path.join(train_dir, cat, img))
    for img in test_images:
        shutil.copy(os.path.join(cat_dir, img), os.path.join(test_dir, cat, img))

print("✅ Dataset split complete! Check dataset/train and dataset/test folders.")