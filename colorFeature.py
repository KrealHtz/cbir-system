# import the necessary packages
import numpy as np
import cv2

class ColorDescriptor:
    def __init__(self, bins):
        # 存储 3D 直方图的数量
        self.bins = bins

    def describe(self, image):
        # 将图像转换为 HSV 色彩空间并初始化
        # 用于量化图像的特征
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        features = []
        # 获取尺寸并计算图像的中心
        (h, w) = image.shape[:2]
        (cX, cY) = (int(w * 0.5), int(h * 0.5))
        # 将图像分成四份 rectangles/segments (top-left,top-right, bottom-right, bottom-left)
        segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h), (0, cX, cY, h)]
        # 构建代表图像中心的椭圆蒙版
        (axesX, axesY) = (int(w * 0.75) // 2, int(h * 0.75) // 2)
        ellipMask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)
        # loop over the segments
        for (startX, endX, startY, endY) in segments:
            # 为图像的每个角构建一个掩码，从中减去椭圆中心
            cornerMask = np.zeros(image.shape[:2], dtype="uint8")
            cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
            cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)
            # 从图像中提取颜色直方图，然后更新特征向量
            hist = self.histogram(image, cornerMask)
            features.extend(hist)
        # 从椭圆区域提取颜色直方图并更新特征向量
        hist = self.histogram(image, ellipMask)
        features.extend(hist)
        # 返回特征向量
        return features

    def histogram(self, image, mask):
        # 使用提供的每个通道的 bin 数量，从图像的遮罩区域中提取 3D 颜色直方图
        hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins, [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        # 返回直方图
        return hist





