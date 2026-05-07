from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
import random
import torch
import numpy as np
import torch.nn as nn
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
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Step 2 Pulls in the images and uses the prior tansform to turn the images into vectors

mndataNum = MNIST('./images/numbers')
mndataChar = MNIST('./images/letters')

images_raw, labels_raw = mndataChar.load_training()
char_images, char_labels = transform_mnist_data(images_raw, labels_raw, is_letters=True)

nums_raw, nums_labels_raw = mndataNum.load_training()
num_images, num_labels = transform_mnist_data(nums_raw, nums_labels_raw, is_letters=False)

import matplotlib.pyplot as plt

def visualize_tensor(tensor_img, label, is_letters=False):
    """
    Displays a single image from the transformed tensor.
    """
    # 1. Remove the 'Batch' and 'Channel' dimensions so it's just (28, 28)
    # .squeeze() turns (1, 1, 28, 28) into (28, 28)
    img = tensor_img.squeeze().numpy()

    # 2. Plot the image
    plt.imshow(img, cmap='gray')
    
    # 3. Format the title
    if is_letters:
        # Add 1 back to the label (0-25 -> 1-26) then convert to A-Z
        title = f"Letter: {chr(int(label) + 65)}"
    else:
        title = f"Number: {int(label)}"
        
    plt.title(title)
    plt.axis('off') # Hide the X/Y coordinates
    plt.show()

# idx = random.randrange(0, len(char_images))
visualize_tensor(char_images[0], char_labels[0], is_letters=True)
visualize_tensor(char_images[2], char_labels[2], is_letters=True)
visualize_tensor(char_images[3], char_labels[3], is_letters=True)
visualize_tensor(char_images[4], char_labels[4], is_letters=True)
visualize_tensor(char_images[5], char_labels[5], is_letters=True)
visualize_tensor(char_images[6], char_labels[6], is_letters=True)


# --- Check a Number ---
# idx_n = random.randrange(0, len(num_images))
visualize_tensor(num_images[0], num_labels[0], is_letters=False)
visualize_tensor(num_images[2], num_labels[2], is_letters=False)
visualize_tensor(num_images[3], num_labels[3], is_letters=False)
visualize_tensor(num_images[4], num_labels[4], is_letters=False)
visualize_tensor(num_images[5], num_labels[5], is_letters=False)
visualize_tensor(num_images[6], num_labels[6], is_letters=False)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Step 3 Sets up device to use the gpu if avalible and cuda then seeding if gpu is available so the resaults stay the same


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Step 4 Creating the model itself with what each layer does and how


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Step 5 pulls model down and puts it in the cpu/gpu and determine what algorithm to use


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Step 6 detirmine how many loops you run through and pulling data and putting it into the model to train


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Step 7 takes resaults from training and testing to see how accurate the model is using data the model hasn't seen

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Step 8 Saveing trained data