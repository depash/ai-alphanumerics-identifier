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

mndata = MNIST('./images')

images, labels = mndata.load_training()

index = random.randrange(0, len(images))