from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from mnist import MNIST

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Step 1 make a transfom that will be used to turn the images into vectores and it will resize,convert to tensor, then normalize

def transform_mnist_data(images, labels=None, is_letters=False):
    tensor_img = torch.tensor(images).view(-1, 1, 28, 28).float()

    tensor_img /= 255.0

    if is_letters:
        tensor_img = tensor_img.transpose(2, 3)

    if labels is not None:
        tensor_labels = torch.tensor(labels)
        
        if is_letters:
            tensor_labels = tensor_labels - 1
            
        return tensor_img, tensor_labels

    return tensor_img

augmentation_pipeline = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5)
])

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Step 2 Pulls in the images and uses the prior tansform to turn the images into vectors

mndataNum = MNIST('./images/numbers')
mndataChar = MNIST('./images/letters')

images_raw, labels_raw = mndataChar.load_training()
nums_raw, nums_labels_raw = mndataNum.load_training()

test_nums_raw, test_nums_labels_raw = mndataNum.load_testing()
test_chars_raw, test_chars_labels_raw = mndataChar.load_testing()

num_images, num_labels = transform_mnist_data(nums_raw, nums_labels_raw, is_letters=False)
char_images, char_labels = transform_mnist_data(images_raw, labels_raw, is_letters=True)

test_char_images, test_char_labels = transform_mnist_data(test_chars_raw, test_chars_labels_raw, is_letters=True)
test_num_images, test_num_labels = transform_mnist_data(test_nums_raw, test_nums_labels_raw, is_letters=False)

combined_char_labels = char_labels + 10
combined_test_char_labels = test_char_labels + 10

all_images = torch.cat((num_images, char_images), dim=0)
all_labels = torch.cat((num_labels, combined_char_labels), dim=0)
all_test_images = torch.cat((test_num_images, test_char_images), dim=0)
all_test_labels = torch.cat((test_num_labels, combined_test_char_labels), dim=0)

full_dataset = TensorDataset(all_images, all_labels)

train_loader = DataLoader(
    full_dataset, 
    batch_size=64, 
    shuffle=True, 
    num_workers=0 
)

test_dataset = TensorDataset(all_test_images, all_test_labels)

test_loader = DataLoader(
    test_dataset, 
    batch_size=64, 
    shuffle=False,
    num_workers=0
)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Step 3 Sets up device to use the gpu if avalible and cuda then seeding if gpu is available so the resaults stay the same

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Step 4 Creating the model itself with what each layer does and how

class alphanumericsCNN(nn.Module):
    def __init__(self, num_classes=43):
        super(alphanumericsCNN, self).__init__()
        self.backbone = models.resnet18(num_classes=num_classes)

        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()

    def forward(self, x):
        return self.backbone(x)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Step 5 pulls model down and puts it in the cpu/gpu and determine what algorithm to use

model = alphanumericsCNN().to(device)
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Step 6 detirmine how many loops you run through and pulling data and putting it into the model to train

num_epochs = 15

for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0

    model.train()
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        images = augmentation_pipeline(images)

        # images_224 = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total

    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Step 7 takes resaults from training and testing to see how accurate the model is using data the model hasn't seen

model.eval()
val_loss = 0.0
val_correct = 0
val_total = 0

with torch.no_grad():
    for images,labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        # images_224 = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)

        outputs = model(images)

        loss = criterion(outputs, labels)

        val_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        val_total += labels.size(0)

        val_correct += (predicted == labels).sum().item()

val_loss /= len(test_loader)

val_acc = 100 * val_correct / val_total

print(
    f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% | "
    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Step 8 Saveing trained data

torch.save(model.state_dict(), "alphanumerics_model.pth")

# Epoch 1/15 | Loss: 0.4378 | Acc: 87.28%
# Epoch 2/15 | Loss: 0.1863 | Acc: 93.76%
# Epoch 3/15 | Loss: 0.1599 | Acc: 94.59%
# Epoch 4/15 | Loss: 0.1445 | Acc: 95.00%
# Epoch 5/15 | Loss: 0.1352 | Acc: 95.31%
# Epoch 6/15 | Loss: 0.1281 | Acc: 95.53%
# Epoch 7/15 | Loss: 0.1236 | Acc: 95.65%
# Epoch 8/15 | Loss: 0.1165 | Acc: 95.87%
# Epoch 9/15 | Loss: 0.1126 | Acc: 95.98%
# Epoch 10/15 | Loss: 0.1090 | Acc: 96.09%
# Epoch 11/15 | Loss: 0.1074 | Acc: 96.16%
# Epoch 12/15 | Loss: 0.1034 | Acc: 96.24%
# Epoch 13/15 | Loss: 0.1002 | Acc: 96.36%
# Epoch 14/15 | Loss: 0.0989 | Acc: 96.37%
# Epoch 15/15 | Loss: 0.0972 | Acc: 96.47%
# Train Loss: 0.0972 | Train Acc: 96.47% | Val Loss: 0.0960 | Val Acc: 96.64%