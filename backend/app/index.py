from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class alphanumericsCNN(nn.Module):
    def __init__(self, num_classes=36):
        super(alphanumericsCNN, self).__init__()
        self.backbone = models.resnet18(num_classes=num_classes)

        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()

    def forward(self, x):
        return self.backbone(x)

model = alphanumericsCNN()
model.load_state_dict(
    torch.load("models/alphanumerics_model.pth", map_location="cpu")
)
model.eval()

# classes = [

# ]