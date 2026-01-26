import os
import matplotlib.pyplot as plt
import numpy as np

if not os.path.exists("MINIST_dataset"):
    os.system("git clone git@github.com:DeepTrackAI/MNIST_dataset.git")

train_path = os.path.join("MNIST_dataset", "mnist", "train")
print("train_path: ", train_path)
train_images_files = sorted(os.listdir(train_path))
print(len(train_images_files))

train_images = []
for file in train_images_files:
    image = plt.imread(os.path.join(train_path, file))
    train_images.append(image)
print(len(train_images))
print(train_images[0].shape)

train_digits = []
for file in train_images_files:
    filename = os.path.basename(file)
    digit = int(filename[0])
    train_digits.append(digit)

fig, axs = plt.subplots(nrows=3, ncols=10, figsize=(20, 6))
for ax in axs.ravel():
    idx_image = np.random.choice(60000)
    ax.imshow(train_images[idx_image], cmap="Greys")
    ax.set_title(f"Label: {train_digits[idx_image]}", fontsize=20)
    ax.axis("off")
plt.show()
