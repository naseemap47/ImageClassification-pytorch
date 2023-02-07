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
from PIL import Image
import pandas as pd


class_names = os.listdir('data/train')
class_names.sort()
print(class_names)

num_class = len(class_names)

checkpoint = torch.load('best_checkpoint.model')
model = ConvNet(num_class)
model.load_state_dict(checkpoint)
print('Loaded Saved Model...')
print(model.eval())

# Data-preprocessing
transformer = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(), # 0-255 to 0-1, numpy to tensors
    transforms.Normalize([0.5, 0.5, 0.5],
                          [0.5, 0.5, 0.5])
])


def get_predition(img_path, transformer):
    img = Image.open(img_path)
    img_tensor = transformer(img).float()
    img_tensor = img_tensor.unsqueeze_(0)
    if torch.cuda.is_available():
        img_tensor.cuda()
    input_img = Variable(img_tensor)
    pred_out = model(input_img)
    pred_id = pred_out.data.numpy().argmax()
    pred = class_names[pred_id]
    return pred

pred_dict = []
img_list = glob.glob('data/pred/*.jpg')
for img_path in img_list:
        
    pred = get_predition(img_path, transformer)
    img_path = os.path.split(img_path)[1]
    pred_dict.append([img_path, pred])

df = pd.DataFrame(pred_dict, columns=['Image', 'Prediction'])
print(df)
df.to_csv('result.csv', index=False)
