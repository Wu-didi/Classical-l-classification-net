#-*- codeing = utf-8 -*- 
#@Time: 2021/4/19 13:31
#@Author : dapao
#@File : predict.py
#@Software: PyCharm

import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import AlexNet

def main(num_classes):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ]
    )

    #load image
    img_path = r"D:\classic_nets\data_set\flower_data\val\daisy\253426762_9793d43fcd.jpg"
    assert os.path.exists(img_path),"file:'{}’ does not exist" .format(img_path)
    # assert  检查条件，不符合就终止程序

    img = Image.open(img_path)
    plt.imshow(img)
    img = data_transform(img)
    #expand batch dimension
    img = torch.unsqueeze(img,dim=0)#添加一个维度，网络输入的是一个四维的

    #read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path),"file: '{}' does not exist.".format(json_path)

    json_file = open(json_path,'r')
    class_indict = json.load(json_file)

    #creat model
    model = AlexNet(num_classes = num_classes).to(device)

    #load model weights
    weights_path = "./AlexNet.pth"
    assert os.path.exists(weights_path),"file:weights path doesn't exist"
    model.load_state_dict(torch.load(weights_path))

    model.eval()
    with torch.no_grad():
        #predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output,dim=0)
        predict_cla = torch.argmax(predict).numpy()


    print_result = "class:{} prob:{:.3f}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_result)
    print(print_result)
    plt.show()

if __name__ == '__main__':
    main(num_classes=5)
