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
from model import ConvNet


BATCH_SIZE = 12
EPOCHS = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Data-preprocessing
transformer = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), # 0-255 to 0-1, numpy to tensors
    transforms.Normalize([0.5, 0.5, 0.5],
                          [0.5, 0.5, 0.5])
])

# DataLoader
path_to_train_dir = 'data/train'
path_to_test_dir = 'data/test'

train_loader = DataLoader(
    torchvision.datasets.ImageFolder(path_to_train_dir, transform=transformer),
    batch_size=BATCH_SIZE, shuffle=True
)

test_loader = DataLoader(
    torchvision.datasets.ImageFolder(path_to_test_dir, transform=transformer),
    batch_size=BATCH_SIZE, shuffle=True
)

# Classes
class_names = os.listdir(path_to_train_dir)
class_names.sort()
print(class_names)

# Load custom Model
model = ConvNet(num_classes=len(class_names)).to(device)

# Optimzer
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
loss_function = nn.CrossEntropyLoss()

# Calculate Train and Test data size
train_size = len(glob.glob('data/train/**/*.jpg'))
test_size = len(glob.glob('data/test/**/*.jpg'))

print('Train Data Size: ', train_size)
print('Test Data Size: ', test_size)


# Model Training
print('Training Started.....')
best_accuracy = 0.0
for epoch in range(EPOCHS):
    # eval and train on training dataset
    model.train()
    train_accuracy = 0.0
    train_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
            
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.cpu().data*images.size(0)
        _, prediction = torch.max(outputs.data, 1)

        train_accuracy += int(torch.sum(prediction==labels.data))
    
    train_accuracy = train_accuracy/train_size
    train_loss = train_loss/train_size
    
    # Evaluation on Test Dataset
    model.eval()

    test_accuracy = 0.0
    for i, (images, labels) in enumerate(test_loader):
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

        outputs = model(images)
        _, prediction = torch.max(outputs.data, 1)

        test_accuracy += int(torch.sum(prediction==labels.data))
    
    test_accuracy = test_accuracy/test_size

    print(f'Epoch: {epoch}\nTrain Loss: {train_loss} Train Accuracy: {train_accuracy}\nTest Accuracy: {train_accuracy}')

    # Save best Model
    if test_accuracy > best_accuracy:
        torch.save(model.state_dict(), 'best_checkpoint.model')
        best_accuracy = test_accuracy
