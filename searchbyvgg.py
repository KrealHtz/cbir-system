

from VGGNET import VGGNet

import numpy as np
import h5py

# 打开h5文件
h5f = h5py.File("Feature Library/index.h5", 'r')
# 所有特征
feats = h5f['dataset_1'][:]
# print(feats)
# 所有图像地址
imgNames = h5f['dataset_2'][:]
# print(imgNames)
# 关闭文件
h5f.close()
# 存储相似度得分
score = []
print("--------------------------------------------------")
print("  程序启动........")
print("--------------------------------------------------")


def searchByVgg(image_path):
    print("--------------------------------------------------")
    print("  正在检索........")
    print("--------------------------------------------------")

    # 初始化 VGGNet16 模型
    model = VGGNet()

    # 提取查询图像的特征
    queryVec = model.get_feat(image_path)

    # # 使用冒泡排序线性搜索(性能太差)
    # scores = np.dot(queryVec, feats.T)  # T转置,类似numpy.transpose         矩阵的点积
    # scores2 = list(scores.copy())
    # for k in range(len(scores)):
    #     for j in range(0, len(scores) - k - 1):
    #         if scores[j] < scores[j + 1]:
    #             scores[j], scores[j + 1] = scores[j + 1], scores[j]
    # rank_ID = [scores2.index(s) for s in scores]
    # results = [imgNames[index] for i, index in enumerate(rank_ID[0:10])]
    # #

    # 使用np.argsort()进行线性搜索
    # 计算余弦相似度
    scores = np.dot(queryVec, feats.T)  # T转置,类似numpy.transpose，矩阵的点积
    # print(scores)
    rank_ID = np.argsort(scores)[::-1]
    rank_score = scores[rank_ID]
    # print (rank_ID)
    # 要显示的检索到的图像数量
    res_num = 10
    # 将相似度得分存储以便显示
    score.clear()
    for sc in rank_score[:res_num]:
        score.append(sc)
    # 将最为匹配的几个图片的地址返回
    results = [imgNames[index] for i, index in enumerate(rank_ID[0:res_num])]
    print("最匹配的 %d 张图片为: " % res_num, results)
    return results


# 返回相似度得分
def getScores():
    # print(score)
    return score
