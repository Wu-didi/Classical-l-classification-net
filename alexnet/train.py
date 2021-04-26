#-*- codeing = utf-8 -*- 
#@Time: 2021/4/18 23:10
#@Author : dapao
#@File : train.py
#@Software: PyCharm

import os
import json
import torch
import torch.nn as nn
from torchvision import transforms,datasets,utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from model import AlexNet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([
         transforms.RandomResizedCrop(224),
         transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]),
         "val": transforms.Compose([transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])}
    data_root = os.path.abspath(os.path.join(os.getcwd(),"../.."))
    #data_root = D:\classic_nets
    image_path = os.path.join(data_root,"data_set","flower_data")
    print(image_path)
    assert os.path.exists(image_path),"{} path does not exit.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path,"train"),
                                         transform = data_transform["train"])
    train_num = len(train_dataset)
    print(train_num)

    flower_list = train_dataset.class_to_idx
    #flower_list  {'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflowers': 3, 'tulips': 4}
    cla_dict = dict((val,key) for key,val in flower_list.items())
    #print(cla_dict)#{0: 'daisy', 1: 'dandelion', 2: 'roses', 3: 'sunflowers', 4: 'tulips'}
    #write dict into json file
    json_str = json.dumps(cla_dict,indent=4)
    with open("class_indices.json",'w') as json_file:
        json_file.write(json_str)

    batch_size = 32
    nw = min([os.cpu_count(),batch_size if batch_size>1 else 0,8])
    print("using {} dataloader workers every process".format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = batch_size,
                                               shuffle = True,
                                               num_workers = 0)
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path,"val"),
                                            transform = data_transform["val"])

    val_num = len(validate_dataset)
    print(val_num)#val_num = 364
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size = 4,
                                                  shuffle=False,
                                                  num_workers = 0)
    print("using {} images for training, {} images for validation .".format(train_num,val_num))





    net = AlexNet(num_classes = 5,init_weights=True)
    net.to(device)
    #当我们指定了设备之后，就需要将模型加载到相应设备中，此时需要使用model=model.to(device)，将模型加载到相应的设备中
    loss_function = nn.CrossEntropyLoss()
    #损失函数
    optimizer = optim.Adam(net.parameters(),lr=0.0002)
    #训练所有的参数，学习率为0.0002

    epochs = 10
    save_path = './AlexNet.pth'
    best_acc = 0.0
    train_steps = len(train_loader)
    for epoch in range(epochs):
        #train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        print(train_bar)
        for step,data in enumerate(train_bar):
            images,labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs,labels.to(device))
            loss.backward()
            optimizer.step()
            '''
            总得来说，这三个函数的作用是先将梯度归零（optimizer.zero_grad()），
            然后反向传播计算得到每个参数的梯度值（loss.backward()），
            最后通过梯度下降执行参数更新（optimizer.step()）
            '''


            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch +1,
                                                                     epochs,
                                                                     loss)


        #validate
        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images,val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs,dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
        val_accurate = acc/val_num
        print('[epoch %d] train_loss:%.3f val_accuracy:%.3f'%
              (epoch+1,running_loss/train_steps,val_accurate))



        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(),save_path)

    print("Finished Training")




if __name__ =='__main__':
    main()

