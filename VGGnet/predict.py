#-*- codeing = utf-8 -*- 
#@Time: 2021/4/25 21:35
#@Author : dapao
#@File : predict.py
#@Software: PyCharm


import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from model import vgg


def main(image_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cup")

    data_transform = transforms.Compose(
        [transforms.Resize((224,224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
    )

    img = Image.open(image_path)
    plt.imshow(img)
    img = data_transform(img)
    img = torch.unsqueeze(img,dim=0)

    json_path = "./class_index.json"
    json_file = open(json_path,"r")
    class_indict = json.load(json_file)
    print(class_indict)

    model = vgg(model_name = 'vgg16',num_classes = 5).to(device)

    weight_path = './vgg16Net.pth'
    model.load_state_dict(torch.load(weight_path))


    model.eval()
    with torch.no_grad():
        print(model(img.to(device)))
        output = torch.squeeze(model(img.to(device))).cpu()
        print(output)
        predict = torch.softmax(output,dim=0)
        print(predict)
        predict_cla = torch.argmax(predict).numpy()
        print(predict_cla)

if __name__ == '__main__':
    while True:
        img = input('Input image path:')
        main(img)



