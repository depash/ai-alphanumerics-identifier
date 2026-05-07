from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
import random
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from mnist import MNIST

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Step 1 make a tranfom that will be used to turn the images into vectores and it will resize,convert to tensor, then normalize

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Step 2 Pulls in the images and uses the prior tansform to turn the images into vectors


mndataNum = MNIST('./images/numbers')
mndataChar = MNIST('./images/letters')

images, labels = mndataChar.load_training()

readable_labels = [chr(64 + l) for l in labels]

index = random.randrange(0, len(images))

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