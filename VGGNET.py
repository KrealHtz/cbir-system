import numpy as np
from numpy import linalg as LA
# keras封装了VGG16模型，可直接调用以使用
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input


class VGGNet:
    def __init__(self):
        # weights: 权重，这里选择在ImageNet上预训练的权重
        # pooling: 池化层
        # input_shape: 输入形状
        self.input_shape = (224, 224, 3)
        self.weight = 'imagenet'
        self.pooling = 'max'
        # VGG16()函数的参数意义
        # include_top: 是否包括网络顶部的3个全连接层。
        # weights:  1. 缺省：随机初始化
        #           2. 'imagenet'： 在 ImageNet 上进行预训练的权重
        #           3. 要加载的权重文件的路径
        # input_tensor: 可选的Keras张量(即layers.Input()的输出) 用作模型的图像输入
        # input_shape: 可选的shape元组，只在include_top为False时指定(否则输入的形状必须是(224,224,3)(channels_last数据格式)
        #               或(3,224,224)(channels_first数据格式)。它应该有3个输入通道，宽度和高度应该不小于32。例如(200,200,3)将是一个有效值。
        # pooling: 当include_top为False时，特征提取的可选池模式。
        #           - None表示模型的输出将是最后一个卷积块的4D张量输出。
        #           - avg表示将全局平均池应用于最后一个卷积块的输出，因此模型的输出将是一个2D张量。
        #           - Max意味着全局最大池将被应用。
        # classes: 可选的图像分类数量，仅当 include_top 为 True 时指定，并且如果没有指定 weights 参数。
        # classifier_activation: str或callable。激活功能使用在“顶部”层。仅当include_top = True时需要考虑。
        #               设置classifier_activation=None返回顶层的日志。
        #               当加载预训练的权重时，classifier_activation只能为None或"softmax"。
        #
        # 返回一个 keras.Model 实例
        self.model = VGG16(include_top=False,
                           weights=self.weight,
                           input_shape=(self.input_shape[0], self.input_shape[1], self.input_shape[2]),
                           pooling=self.pooling)
        # 模型概率预测设定
        self.model.predict(np.zeros((1, 224, 224, 3)))

    # 使用vgg16模型提取特征输出归一化特征向量
    def get_feat(self, img_path):
        # 加载图像，格式为PIL，目标尺寸为224*224
        img = image.load_img(img_path,
                             target_size=(self.input_shape[0], self.input_shape[1]))
        # 将 PIL Image 实例转换为 Numpy 数组
        img = image.img_to_array(img)
        # 展开图像数组
        img = np.expand_dims(img, axis=0)
        # 预处理
        img = preprocess_input(img)
        # 返回图像属于每一个类别的概率
        features = self.model.predict(img)
        # 归一化
        norm_feature = features[0] / LA.norm(features[0])
        # print("feature[0]:")
        # print(LA.norm(features[0]))
        return norm_feature
