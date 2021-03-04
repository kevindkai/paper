###-------- 1、手写汉字图像去噪
import cv2
import matplotlib.pyplot as plt
image = cv2.imread(image_path)
img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
#均值滤波
img_m=cv2.medianBlur(img,3)
#中值滤波
img_B=cv2.blur(img,(5,5))
#高斯滤波
img_G=cv2.GaussianBlur(img,(7,7),0) 
#双边滤波
img_F=cv2.bilateralFilter(img,40,75,75)
#效果图展示,下同
plt.rcParams['font.sans-serif']=['SimHei'] #解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(8,12))
plt.xlabel("图像去噪结果示意图（部分）")
plt.subplot(1,5,1)
plt.title("原始图")
plt.axis("off")
plt.imshow(img)
plt.subplot(1,5,2)
plt.title("均值滤波")
plt.axis("off")
plt.imshow(img_m)
plt.subplot(1,5,3)
plt.title("中值滤波")
plt.axis("off")
plt.imshow(img_B)
plt.subplot(1,5,4)
plt.title("高斯滤波")
plt.axis("off")
plt.imshow(img_G)
plt.subplot(1,5,5)
plt.title("双边滤波")
plt.axis("off")
plt.imshow(img_F)
plt.show()

###-------- 2、手写汉字图像灰度化
#最大值灰度化
def gray_max_rgb(inputimagepath):
    img = cv2.imread(inputimagepath)
    gray_max_rgb_image = img.copy()
    img_shape = img.shape
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            gray_max_rgb_image[i,j] = max(img[i,j][0],img[i,j][1],img[i,j][2])
#平均值灰度化
def gray_mean_rgb(inputimagepath):
    img = cv2.imread(inputimagepath)
    gray_mean_rgb_image = img.copy()
    img_shape = img.shape
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            gray_mean_rgb_image[i,j] = (int(img[i,j][0])+int(img[i,j][1])+int(img[i,j][2]))/3
#加权平均灰度化
def gray_weightmean_rgb(wr,wg,wb,inputimagepath):
    img = cv2.imread(inputimagepath)
    gray_weightmean_rgb_image = img.copy()
    img_shape = img.shape
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            gray_weightmean_rgb_image[i,j] = (int(wr*img[i,j][2])+int(wg*img[i,j][1])+int(wb*img[i,j][0]))/3
            
###-------- 3、手写汉字图像反二值化 
#全局阈值反二值化
ret,thresh=cv2.threshold(image,160,255,cv2.THRESH_BINARY_INV)
#局部阈值反二值化
ret1,thresh1 = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)  
#动态阈值反二值化  
ret2,thresh2=cv2.adaptiveThreshold(image,0,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,4,2) 
