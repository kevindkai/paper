###1、导入所需的函数包
import torch
import torchvision
from torchvision import datasets, models, transforms
import os
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch.utils.data import SubsetRandomSampler
import time
import csv 
import numpy as np
%matplotlib inline
### 2、数据准备
#将图像数据转化为tensor
data_transform = transforms.Compose([transforms.Scale([224,224]),
                                        transforms.Resize([224,224]),
                                        transforms.ColorJitter(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])])

image_datasets = datasets.ImageFolder(root ="/train_data",transform = data_transform)
batch_size=16
index_classes = image_datasets.class_to_idx
# 数据集的划分，分为训练集和验证集
validation_split = .01
shuffle_dataset = True
random_seed= 42

# Creating data indices for training and validation splits:
dataset_size = len(image_datasets)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
# 数据集划分
validation_split = .1
shuffle_dataset = True
random_seed= 42
# Creating data indices for training and validation splits:
dataset_size = len(image_datasets)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices_1,val_indices_1 = indices[split:],indices[:split]
# Creating PT data samplers and loaders:
train_data = SubsetRandomSampler(train_indices_1)
valid_data = SubsetRandomSampler(val_indices_1)

train_loader = torch.utils.data.DataLoader(image_datasets, batch_size=batch_size, 
                                           sampler=train_data)
val_loader = torch.utils.data.DataLoader(image_datasets, batch_size=batch_size,
                                                sampler=valid_data)

dataloader={
    "train":train_loader,
    "valid":val_loader
}
img_datasets={
    "train":train_data,
    "valid":valid_data
}
###3、手写汉字识别模型的搭建与训练
#（1）AlexNet-C模型的搭建
model = models.alexnet(pretrained=True)
Use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
print(model)
for parma in model.parameters():
    parma.requires_grad = True
    model.classifier = torch.nn.Sequential(torch.nn.Dropout(p=0.5,inplace=False),
                                           torch.nn.Linear(9216, 4096),
                                           torch.nn.ReLU(inplace=True),
                                           torch.nn.Dropout(p=0.5,inplace=False),
                                           torch.nn.Linear(4096,2048),
                                           torch.nn.ReLU(inplace=True),
                                           torch.nn.Linear(2048, 473))
model.to(device)
#（2）GoogLeNet-C模型的搭建
model = models.googlenet(pretrained=True)
Use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
print(model)
for parma in model.parameters():
    parma.requires_grad = True
    model.fc = torch.nn.Sequential(torch.nn.Linear(1000,473))
model.to(device)
#（3）ResNet-C模型的搭建
model = models.resnet152(pretrained=True)
Use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
print(model)
for parma in model.parameters():
    parma.requires_grad = True
    model.fc = torch.nn.Sequential(torch.nn.Linear(2048,1024),
                                   torch.nn.Dropout(p=0.5,inplace=False),
                                   torch.nn.Linear(1024,473))
model.to(device)
#（4）DenseNet-C模型的搭建
for parma in model.parameters():
    parma.requires_grad = True
    model.classifier = torch.nn.Sequential(torch.nn.Linear(1920, 1024),
                                           torch.nn.ReLU(inplace=True),
                                           torch.nn.Linear(1024, 473))
model.to(device)
#损失函数和优化器
cost = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
loss_f = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.00001)
epoch_n = 20
#模型训练
time_open = time.time()
for epoch in range(epoch_n):
    print("Epoch {}/{}".format(epoch, epoch_n - 1))
    print("-"*10)
    
    for phase in ["train", "valid"]:
        if phase == "train":
            print("Training...")
            model.train(True)
        else:
            print("Validing...")
            model.train(False)
            
        running_loss = 0.0
        running_corrects = 0
    
        for batch, data in enumerate(dataloader[phase], 1):
            X, y = data
            if Use_gpu:
                X, y = Variable(X.cuda()), Variable(y.cuda())
            else:
                X, y = Variable(X), Variable(y)
            X = X.to(device)
            y = y.to(device)
        
            y_pred = model(X)
            _, pred = torch.max(y_pred.data, 1)
            optimizer.zero_grad()
            loss = loss_f(y_pred, y)
        
            if phase == "train":
                loss.backward()
                optimizer.step()
            
            running_loss += loss.data
            running_corrects += torch.sum(pred == y.data)
               
            if batch%500 == 0 and phase =="train":
                print("Batch {}, Train Loss:{:.4f}, Train ACC:{:.4f}%".format(batch, running_loss//batch, 100*running_corrects//(16*batch)))

        epoch_loss = running_loss*16//len(img_datasets[phase])
        epoch_acc = 100*running_corrects//len(img_datasets[phase])
        print("{} Loss:{:.4f} Acc:{:.4f}%".format(phase, epoch_loss,epoch_acc))
            
time_end = time.time() - time_open
print(time_end)
# 将训练好的模型进行保存
torch.save(model,"model_name.pkl")
###4、模型测试
#加载训练好的模型和测试数据集
model = torch.load('/model_name.pkl') 
test_datasets = datasets.ImageFolder(root ="/test_data",transform = data_transform)
test_loader=torch.utils.data.DataLoader(test_datasets,batch_size=1,shuffle=False)
def Name(i):
    name = test_datasets.imgs[i][0].split("/")[-2]
    return name
Use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
#将测试集代入测试
import json
print("model_name模型的预测情况如下：")
for i, data in enumerate(test_loader):
    classes=[]
    probality=[]
    X, y = data
    X, y = Variable(X.cuda()), Variable(y.cuda())
    X = X.to(device)
    y = y.to(device)
    y_pred = model(X)
    y_1 = y_pred.data.cpu().numpy()
    y_e = np.exp(y_1)
    y_num = y_e.sum()
    maxk=max((1,3))
    y_resize = y.view(-1,1)
    _,pred=y_pred.topk(maxk,1,True,True)
    pr = _.data.cpu().numpy()
    pr_e = np.exp(pr)
    prob = pr_e/y_num
    prob = prob[0]
    for n in prob:
        probality.append(n)
    predict=pred.data.cpu().numpy()
    num = _.data.cpu().numpy()
    for m in predict[0]:
        M = list (index_classes.keys()) [list (index_classes.values()).index (m)]
        classes.append(M)
    #输出预测结果
    print("{}的".format(Name(i))+"top3预测的类别为：",classes,"相应的概率为：",probality,sep=",")
 #   print("{}的".format(Name(i))+"top3的概率为：",probality)
    print("-----------------------------------------------------"*2)
    #保存预测结果
    with open("result_name.csv",mode="a+",newline='',encoding='utf-8') as file:
        csv_writer = csv.writer(file, delimiter=',')    # 指定分隔符为逗号
        csv_writer.writerow([Name(i), classes])