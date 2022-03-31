import os
import h5py
import numpy as np
from VGGNET import VGGNet
from ColourMoments import color_moments
import gc
import cv2
import numpy as np
import os
import joblib
from joblib.numpy_pickle_utils import xrange
from scipy.cluster.vq import *
from sklearn import preprocessing
from colorFeature import ColorDescriptor
from GLCM import *
from ShapeHistogram import *
import time
import hashing
start =time.time()
#中间写上代码块




# path = "dataset/"
# print("数据集：")
# print(os.listdir(path))


# 获取数据集所有图片
def getAllPics(lpath = "dataset/"):
    path = lpath
    image_paths = []
    # 获取dataset/下所有数据集文件夹
    folders = os.listdir(path)
    # 遍历每个数据集
    for folder in folders:
        # print(folder)
        # 获取该数据集下所有子文件夹
        folders_1 = os.listdir(os.path.join(path, folder))
        # 遍历每个子文件夹
        for folder_1 in folders_1:
            # 获取所有子文件夹下所有文件
            ls = os.listdir(os.path.join(path, folder + "/", folder_1))
            # 遍历所有文件
            for image_path in ls:
                # 如果是.jpg格式才收录
                if image_path.endswith('jpg'):
                    # 路径连接
                    image_path = os.path.join(path, folder + "/", folder_1 + "/", image_path)
                    # print("正在获取图片   "+image_path)
                    # 存储
                    image_paths.append(image_path)
    # 返回所有图片列表
    return image_paths


def mainget(listget = getAllPics()):
    # # 获取所有图片
    img_list = listget
    print("图片总数量：" + len(img_list).__str__() + "张")
    print("--------------------------------------------------")
    print(" 开始提取特征......   ")
    print("--------------------------------------------------")

    # ##################### VGG-16 #################################
    # features = []
    # names = []
    #
    # model = VGGNet()
    # for i, img_path in enumerate(img_list):
    #     norm_feat = model.get_feat(img_path)
    #     img_name = img_path
    #     features.append(norm_feat)
    #     names.append(img_name)
    #     print("正在提取图像特征：第 %d 张 , 共 %d 张......." % ((i + 1), len(img_list)) + img_name)
    #
    # feats = np.array(features)
    # # print(feats)
    # # 用于存储提取特征的文件
    # output = "index.h5"
    #
    # print("--------------------------------------------------")
    # print(" 正在将提取到的特征数据存储到文件中......")
    # print("--------------------------------------------------")
    #
    # h5f = h5py.File(output, 'w')
    # h5f.create_dataset('dataset_1', data=features)
    # h5f.create_dataset('dataset_2', data=np.string_(names))
    # h5f.close()
    # ##################### VGG-16 #################################






    ###################### dhash##################################################

    print("准备提取哈希特征描述符......")


    output = open("Feature Library/dhash.csv", "w")

    print("开始提取哈希特征描述符......")
    for image_path in img_list:
        print("正在提取图像的哈希特征描述符："+image_path)
        # imageID唯一标注图片
        imageID = image_path[image_path.find("dataset"):]
        # print("imageID:"+imageID)
        image = cv2.imread(image_path)
        # 获取特征描述符，并转为list形式
        features = hashing.convert_hash(hashing.dhash(image))
        # 将特征描述符写入索引文件
        # print(features)
        # print(features)
        output.write("%s,%s\n" % (imageID, str(features)))



    print("哈希特征描述符完毕")

    ################################# dhash #############################################


    ###################### phash##################################################

    print("准备提取p哈希特征描述符......")


    output = open("Feature Library/phash.csv", "w")

    print("开始提取p哈希特征描述符......")
    for image_path in img_list:
        print("正在提取图像的哈希特征描述符："+image_path)
        # imageID唯一标注图片
        imageID = image_path[image_path.find("dataset"):]
        # print("imageID:"+imageID)
        image = cv2.imread(image_path)
        # 获取特征描述符，并转为list形式
        features = hashing.convert_hash(hashing.pHash(image))
        # 将特征描述符写入索引文件
        # print(features)
        # print(features)
        output.write("%s,%s\n" % (imageID, str(features)))



    print("p哈希特征描述符完毕")

    ################################# phash #############################################

    ###################### ahash##################################################

    print("准备提取a哈希特征描述符......")

    output = open("Feature Library/ahash.csv", "w")

    print("开始提取a哈希特征描述符......")
    for image_path in img_list:
        print("正在提取图像的哈希特征描述符：" + image_path)
        # imageID唯一标注图片
        imageID = image_path[image_path.find("dataset"):]
        # print("imageID:"+imageID)
        image = cv2.imread(image_path)
        # 获取特征描述符，并转为list形式
        features = hashing.convert_hash(hashing.aHash(image))
        # 将特征描述符写入索引文件
        # print(features)
        # print(features)
        output.write("%s,%s\n" % (imageID, str(features)))

    print("a哈希特征描述符完毕")

    ################################# ahash #############################################






    # ###################### 开始提取颜色矩描述符 #################################
    # print("准备提取颜色特征描述符......")
    # # 打开索引文件进行写入,默认为index.csv
    # output = open("Feature Library/colormoment.csv", "w")
    # print("开始提取图像颜色矩......")
    # for image_path in img_list:
    #     print("正在提取图像的颜色矩："+image_path)
    #     # imageID唯一标注图片
    #     imageID = image_path[image_path.find("dataset"):]
    #     # print("imageID:"+imageID)
    #     # image = cv2.imread(image_path)
    #     # 获取特征描述符，并转为list形式
    #     features = list(np.array(color_moments(image_path)))
    #     # 将特征描述符写入索引文件
    #     # print(features)
    #     features = [str(f) for f in features]
    #     # print(features)
    #     output.write("%s,%s\n" % (imageID, ",".join(features)))
    # print("颜色矩描述符完毕")
    # ################################# 颜色特征描述符完毕 #############################################
    #
    #
    #
    #
    # ###################### 开始提取颜色特征描述符 #################################
    # print("准备提取颜色特征描述符......")
    # # 初始化颜色描述符
    # cd = ColorDescriptor((8, 12, 3))
    # # 打开索引文件进行写入,默认为index.csv
    # output = open("Feature Library/colorhis.csv", "w")
    # print("开始提取图像颜色特征描述符......")
    # for image_path in img_list:
    #     print("正在提取图像的颜色特征描述符："+image_path)
    #     # imageID唯一标注图片
    #     imageID = image_path[image_path.find("dataset"):]
    #     # print("imageID:"+imageID)
    #     image = cv2.imread(image_path)
    #     # 获取特征描述符，并转为list形式
    #     features = list(np.array(cd.describe(image)))
    #     # 将特征描述符写入索引文件
    #     # print(features)
    #     features = [str(f) for f in features]
    #     # print(features)
    #     output.write("%s,%s\n" % (imageID, ",".join(features)))
    # print("颜色特征描述符完毕")
    # ################################# 颜色特征述符完毕 #############################################





    # ################################# 纹理特征 #############################################
    # print("准备提取纹理特征描述符......")
    # # 打开索引文件进行写入,默认为GLCM.csv
    # output = open("Feature Library/GLCM.csv", "w+")
    # print("开始提取纹理特征描述符......")
    # for image_path in img_list:
    #     print("正在提取图像的纹理特征描述符："+image_path)
    #     # imageID唯一标注图片
    #     imageID = image_path[image_path.find("dataset"):]
    #     # print("imageID:"+imageID)
    #     image = cv2.imread(image_path)
    #     # 获取特征描述符，并转为list形式
    #     features = list(np.array(getglcm(image)))
    #     # 将特征描述符写入索引文件
    #     # print(features)
    #     features = [str(f) for f in features]
    #     # print(features)
    #     output.write("%s,%s\n" % (imageID, ",".join(features)))
    # print("纹理特征描述符完毕")
    # ################################# 纹理特征 #############################################





    # ################################# 边缘特征 #############################################
    # print("准备提取边缘特征描述符......")
    # # 打开索引文件进行写入,默认为GLCM.csv
    # output = open("Feature Library/ShapeHis.csv", "w+")
    # print("开始提取边缘特征描述符......")
    # for image_path in img_list:
    #     print("正在提取图像的边缘特征描述符："+image_path)
    #     # imageID唯一标注图片
    #     imageID = image_path[image_path.find("dataset"):]
    #     # print("imageID:"+imageID)
    #     image = cv2.imread(image_path)
    #     # 获取特征描述符，并转为list形式
    #     features = list(np.array(ft(image)))
    #     # 将特征描述符写入索引文件
    #     # print(features)
    #     features = [str(f) for f in features]
    #     # print(features)
    #     output.write("%s,%s\n" % (imageID, ",".join(features)))
    # print("边缘特征描述符完毕")
    # ################################# 边缘特征 #############################################
    #
    #
    #
    #
    #
    #
    #
    #
    #
    # ############################# 准备开始提取所有图片的sift特征   #####################################
    #
    # print("准备提取所有图片的sift特征......")
    #
    # # 设置聚类中心数
    # numWords = 64
    #
    # # 创建特征提取和关键点检测器对象
    # sift_det=cv2.SIFT_create()
    #
    # # 列出所有描述符的存储位置
    # des_list=[]  # 特征描述
    #
    # print("开始提取图像sift特征描述符......")
    # s=0
    # for image_path in img_list:
    #     print("正在提取图像的sift特征描述符："+image_path)
    #     # 读取图片文件
    #     img = cv2.imread(image_path)
    #     # 将图像转换为灰度图
    #     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #     # 检测关键点并计算描述符
    #     kp, des = sift_det.detectAndCompute(gray, None)
    #     des_list.append((image_path, des))
    #
    #
    # # 将所有描述符垂直堆叠在一个 numpy 数组中
    # descriptors = des_list[0][1]
    # print('生成向量数组中......')
    # count=1
    # for image_path, descriptor in des_list[1:]:
    #     print(count)
    #     count+=1
    #     descriptors = np.vstack((descriptors, descriptor))
    #
    # # 执行 k-means clustering
    # print ("开始 k-means 聚类: %d words, %d key points" %(numWords, descriptors.shape[0]))
    # voc, variance = kmeans(descriptors, numWords, 1)
    #
    # # 计算特征的直方图
    # print("计算特征直方图中......")
    # im_features = np.zeros((len(img_list), numWords), "float32")
    # # print(len(image_paths))
    # # for i in range(len(image_paths)):
    # for i in range(len(img_list)):
    #     words, distance = vq(des_list[i][1],voc)
    #     print(i)
    #     for w in words:
    #         im_features[i][w] += 1
    #
    # # 执行 Tf-Idf 矢量化
    # print("进行Tf-Idf 矢量化中......")
    # nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
    # idf = np.array(np.log((1.0*len(img_list)+1) / (1.0*nbr_occurences + 1)), 'float32')
    #
    # # Perform L2 normalization
    # # 执行 L2 规范化
    # print("正在进行归一化处理......")
    # im_features = im_features*idf
    # im_features = preprocessing.normalize(im_features, norm='l2')
    #
    # print('保存词袋模型文件中.......')
    # joblib.dump((im_features, img_list, idf, numWords, voc), "Feature Library/bow.pkl", compress=3)
    #
    # print("sift特征提取完毕！")
    #
    # ################################# sift特征提取结束 #############################################
    #
    # print("特征描述符提取完毕！")




    end = time.time()
    print('Running time: %s Seconds'%(end-start))