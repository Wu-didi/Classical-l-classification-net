#-*- codeing = utf-8 -*- 
#@Time: 2021/4/24 22:14
#@Author : dapao
#@File : train.py
#@Software: PyCharm

import os
import json

import torch
import torch.nn as nn
from torchvision import transforms,datasets
import torch.optim as optim
from tqdm import tqdm
from model import vgg


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using{} device".format(device))

    data_transform = {
        "train":transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]),
        "val": transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
    }

    data_root = os.path.abspath(os.path.join(os.getcwd(),"../.."))
    image_path = os.path.join(data_root,"data_set","flower_data")
    assert  os.path.exists(image_path),"image path does not exit"

    train_dataset = datasets.ImageFolder(root=os.path.join(image_path,"train"),
                                         transform=data_transform["train"])
    print(train_dataset)
    train_num = len(train_dataset)




    #生成一个与
    flow_list = train_dataset.class_to_idx
    print(flow_list)
    cla_dict = dict((val,key) for key,val in flow_list.items())
    json_str = json.dumps(cla_dict,indent=4)
    with open("class_index.json",'w') as json_file:
        json_file.write(json_str)


    batch_size = 2
    nw = [os.cpu_count(),batch_size if batch_size>1 else 0,8]

    print(nw)
    print(min(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = batch_size,
                                               num_workers = 0)
    validate_dataset = datasets.ImageFolder(root = os.path.join(image_path,'val'),
                                            transform=data_transform['val'])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size,
                                                  num_workers=0)
    print("using {} images for training, {} images for validation".format(train_num,val_num))

    model_name = 'vgg16'
    net = vgg(model_name=model_name,num_classes= 5,init_weights=True)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(),lr=0.0001)

    epochs = 2
    best_acc = 0.0
    save_path = './{}Net.pth'.format(model_name)
    train_steps = len(train_loader)
    for epoch in range(epochs):
        net.train()#net.train和net。eval的区别目前的理解是，在train阶段我们会使用dropoout和bn层等操作，
        # 但是在验证集时，我们并不用这两层，利用eval和train来划分这两个
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

            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch,epochs,loss)

        net.eval()
        acc = 0.0
        with torch.no_grad():#在验证阶段，禁止pytorch对参数进行跟踪，在验证阶段不进行梯度的跟踪
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images,val_labes = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs,dim=1)[1]
                acc += torch.eq(predict_y,val_labes.to(device)).sum().item()

        val_accurate = acc/val_num
        print("[epoch %d] train_loss:%.3f val_accuracy:%.3f"%(epoch+1,running_loss/train_steps,val_accurate))

        if val_accurate>best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(),save_path)
    print("Finished Training")
if __name__ == '__main__':
    main()

