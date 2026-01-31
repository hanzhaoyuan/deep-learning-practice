import os
import matplotlib.pyplot as plt
import numpy as np
import deeplay as dl
import deeplay as dl
from torch.nn import ReLU, MSELoss, Sigmoid
from seaborn import cubehelix_palette, heatmap
from torch.nn import Softmax

if not os.path.exists("MNIST_dataset"):
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



test_path = os.path.join("MNIST_dataset", "mnist", "test")
test_images_files = sorted(os.listdir(test_path))

test_images, test_digits = [], []
for file in test_images_files:
    image = plt.imread(os.path.join(test_path, file))
    test_images.append(image)

    filename = os.path.basename(file)
    digit = int(filename[0])
    test_digits.append(digit)

test_images_digits = list(zip(test_images, test_digits))
test_dataloader = dl.DataLoader(test_images_digits, shuffle=False)

trainer = dl.Trainer(max_epochs=1, accelerator="auto")

mlp_template = dl.MultiLayerPerceptron(
    in_features=28*28, hidden_features=[32, 32], out_features=10,
)
mlp_template[..., "activation"].configure(Sigmoid)
mlp_model = mlp_template.create()

# print(mlp_model)
print(f"{sum(p.numel() for p in mlp_model.parameters())} trainable parameters")

classifier_template = dl.Classifier(
    model=mlp_template, num_classes=10, make_targets_one_hot=True,
    loss=MSELoss(), optimizer=dl.SGD(lr=.1)
)
classifier_template[..., "activation#:-1"].configure(ReLU)
classifier_template.configure(optimizer=dl.RMSprop(lr=0.001))
classifier_rmsprop = classifier_template.create()

num_images_x_digit = 3
plt.figure(figsize=(10, num_images_x_digit))
num_fails_x_digit = np.zeros(10, int)
for image, gt_digit in test_dataloader:
    gt_digit = int(gt_digit)
    if num_fails_x_digit[gt_digit] < num_images_x_digit:
        predictions = classifier_rmsprop(image)
        max_prediction, pred_digit = predictions.max(dim=1)
        if pred_digit != gt_digit:
            num_fails_x_digit[gt_digit] += 1
            plt.subplot(num_images_x_digit, 10,
                (num_fails_x_digit[gt_digit] - 1) * 10 + gt_digit + 1)
            plt.imshow(image.squeeze(), cmap="Greys")
            plt.annotate(str(int(pred_digit)), (.8, 1), (1, 1),
                xycoords="axes fraction", textcoords="offset points",
                va="top", ha="left", fontsize=20, color="red")
            plt.axis("off")
    if (num_fails_x_digit >= num_images_x_digit).all():
        break
plt.tight_layout()
plt.show()