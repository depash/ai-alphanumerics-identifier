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

mndataNum = MNIST('./images/numbers')
mndataChar = MNIST('./images/letters')

images, labels = mndataChar.load_training()

readable_labels = [chr(64 + l) for l in labels]

index = random.randrange(0, len(images))