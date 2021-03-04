###1、基于笔画特征的书写质量评判
#导入所需函数包
import os.path
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.autograd import Variable
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings("ignore")
#数据转化
data_transform = transforms.Compose([transforms.Scale([224,224]),
                                        transforms.Resize([224,224]),
                                        transforms.ColorJitter(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])])
img = Image.open(img_path)
img1 = data_transform(img)
#训练好的卷积网络提取笔画特征
model_feature_extractor = torch.load('/model_name.pkl')
model_feature_extractor.fc = nn.Linear(n_dim, n_dim)
torch.nn.init.eye(model_feature_extractor.fc.weight)

for param in model_feature_extractor.parameters():
    param.requires_grad = False

x = Variable(torch.unsqueeze(img1, dim=0).float(), requires_grad=False)
y = model_feature_extractor(x)
features = y.data.numpy()
#书写质量评判
from scipy.spatial.distance import cosine
TR = cosine(features_stand,features_compare)

###2、基于结构特征的书写质量评判
#结构特征提取
def retange(img):
    image=img.astype(np.uint8)
    edged = cv2.Canny(image,20,150)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST,cv2.CHAIN_APPROX_TC89_L1)
    cnts = cnts[1] if  imutils.is_cv2()  else   cnts[0]
    x=[]
    y=[]
    for cnt in cnts:
        ares = cv2.contourArea(cnt)
        if ares<60 and len(cnt)<30:
            continue
        else:
            for k in cnt:
                x.append(k[0][0])
                y.append(k[0][1])
    if x and y :
        x_min=min(x)
        x_max=max(x)
        y_min=min(y)
        y_max=max(y)

        temp=image.copy()
        img1 = temp[y_min:y_max,x_min:x_max]
        
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 255, 255), 1)
   
    else: 
        img1 = image
        print("该图片无法找到外接矩形，请重新调整参数")
    return image,img1         #image是二值图上画白色的矩形框,img1是裁剪的矩形框图
#书写质量评判
def SR(l,L,w,W):
    min_l = min(l,L)
    max_l = max(l,L)
    min_w = min(w,W)
    max_w = max(w,W)
    simi = (min_l/max_l)*(min_w/max_l)
    return simi
    
###3、基于综合特征的书写质量评判
def result(parm):
    result = parm*TR+(1-parm)*SR
    return result