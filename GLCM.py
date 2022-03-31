#!/usr/bin/python
# -*- coding: UTF-8 -*-

# https://blog.csdn.net/kmsj0x00/article/details/79463376

import glob
import cv2
import math

#定义最大灰度级数
gray_level = 32

def maxGrayLevel(img):
    max_gray_level=0
    (height,width)=img.shape
    # print height,width
    for y in range(height):
        for x in range(width):
            if img[y][x] > max_gray_level:
                max_gray_level = img[y][x]
    return max_gray_level+1

def getGlcm(input,d_x,d_y):
    srcdata=input.copy()
    ret=[[0.0 for i in range(gray_level)] for j in range(gray_level)]
    (height,width) = input.shape

    max_gray_level=maxGrayLevel(input)

    #若灰度级数大于gray_level，则将图像的灰度级缩小至gray_level，减小灰度共生矩阵的大小
    if max_gray_level > gray_level:
        for j in range(height):
            for i in range(width):
                srcdata[j][i] = srcdata[j][i]*gray_level / max_gray_level

    for j in range(height-d_y):
        for i in range(width-d_x):
             rows = srcdata[j][i]
             cols = srcdata[j + d_y][i+d_x]
             ret[rows][cols]+=1.0

    for i in range(gray_level):
        for j in range(gray_level):
            ret[i][j]/=float(height*width)

    return ret

def feature_computer(p):
    # con:对比度反应了图像的清晰度和纹理的沟纹深浅。纹理越清晰反差越大对比度也就越大。
    # eng:熵（Entropy, ENT）度量了图像包含信息量的随机性，表现了图像的复杂程度。当共生矩阵中所有值均相等或者像素值表现出最大的随机性时，熵最大。
    # agm:角二阶矩（能量），图像灰度分布均匀程度和纹理粗细的度量。当图像纹理均一规则时，能量值较大；反之灰度共生矩阵的元素值相近，能量值较小。
    # idm:反差分矩阵又称逆方差，反映了纹理的清晰程度和规则程度，纹理清晰、规律性较强、易于描述的，值较大。
    Con=0.0
    Eng=0.0
    Asm=0.0
    Idm=0.0
    for i in range(gray_level):
        for j in range(gray_level):
            Con+=(i-j)*(i-j)*p[i][j]
            Asm+=p[i][j]*p[i][j]
            Idm+=p[i][j]/(1+(i-j)*(i-j))
            if p[i][j]>0.0:
                Eng+=p[i][j]*math.log(p[i][j])
    return Asm,Con,-Eng,Idm

def getglcm(img):

    try:
        img_shape=img.shape
    except:
        print ('imread error')
        return -1
    GLCM_feature = []
    img=cv2.resize(img,(int(img_shape[1]/2),int(img_shape[0]/2)),interpolation=cv2.INTER_CUBIC)

    img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    glcm_0=getGlcm(img_gray, 1,0) # 水平相邻特征0°
    asm,con,eng,idm=feature_computer(glcm_0)
    GLCM_feature.extend([asm,con,eng,idm])
    glcm_1=getGlcm(img_gray, 0,1)
    asm,con,eng,idm=feature_computer(glcm_1) # 90°
    GLCM_feature.extend([asm,con,eng,idm])
    glcm_2=getGlcm(img_gray, 1,1) # 对角相邻 45°
    asm,con,eng,idm=feature_computer(glcm_2)
    GLCM_feature.extend([asm,con,eng,idm])
    # glcm_3=getGlcm(img_gray, -1,1)
    # asm,con,eng,idm=feature_computer(glcm_3)
    # GLCM_feature.extend([asm,con,eng,idm])

    return GLCM_feature