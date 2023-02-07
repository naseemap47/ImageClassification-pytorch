import os
import numpy as np
import torch
import glob
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import torchvision
import pathlib


class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()

        # Output size after convolution filter
        # ((width - kernal_size + 2*padding)/stride) + 1
        

        # input shape = (256, 3, 150, 150)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        # shape = (256, 12, 150, 150)
        self.bn1 = nn.BatchNorm2d(num_features=12)
        # shape = (256, 12, 150, 150)
        self.relu1 = nn.ReLU()
        # shape = (256, 12, 150, 150)
        
        self.pool = nn.MaxPool2d(kernel_size=2)
        # Reduce the image size be factor 2
        # shape = (256, 12, 75, 75)
        
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1)
        # shape = (256, 20, 75, 75)
        self.relu2 = nn.ReLU()
        # shape = (256, 20, 75, 75)

        self.conv3 = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1)
        # shape = (256, 32, 75, 75)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        # shape = (256, 32, 75, 75)
        self.relu3 = nn.ReLU()
        # shape = (256, 32, 75, 75)

        self.fc = nn.Linear(in_features=32*75*75, out_features=num_classes)


    # Feed Forward function
    def forward(self, input):
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu1(output)

        output = self.pool(output)

        output = self.conv2(output)
        output = self.relu2(output)

        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu3(output)
        
        # above output will be in matrix form
        # with shape (256, 32, 75, 75)
        output = output.view(-1, 32*75*75)
        output = self.fc(output)
        return output
        
