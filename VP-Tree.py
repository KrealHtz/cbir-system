from imutils import paths
import pickle
import vptree
import numpy as np
import cv2

# 差异哈希函数
def dhash(image, hashSize=8):
    # 转为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 缩放
    resizeImg = cv2.resize(gray, (hashSize + 1, hashSize))

    # 计算差异值
    diff_value = resizeImg[:, 1:] > resizeImg[:, :-1]
    return sum([2 ** i for (i, v) in enumerate(diff_value.flatten()) if v])


def convert_hash(hash_value):
    # 对差异值处理，返回64位浮点数值
    return int(np.array(hash_value, dtype="float64"))


def hamming(a_dist, b_dist):
    # 计算汉明距离
    return bin(int(a_dist) ^ int(b_dist)).count("1")


hashes = {}

# 换成自己的图像库文件路径
path = "dataset/101_ObjectCategories"
imagePaths = list(paths.list_images(path))
for (i, imagePath) in enumerate(imagePaths):

    print("加载处理图像： {}/{}".format(i + 1,len(imagePaths)))
    image = cv2.imread(imagePath)

    # 计算
    hash_value = dhash(image)
    hash_value = convert_hash(hash_value)

    # 添加进hashes字典
    h = hashes.get(hash_value, [])
    h.append(imagePath)
    hashes[hash_value] = h

# 建vp树
print("构建 VP-Tree...")
points = list(hashes.keys())
tree = vptree.VPTree(points, hamming)

# 保存
print("保存 VP-Tree...")
f = open("Feature Library/vptree.pickle", "wb")
f.write(pickle.dumps(tree))
f.close()

# 保存哈希值
print("保存 hashes...")
f = open("Feature Library/hashes.pickle", "wb")
f.write(pickle.dumps(hashes))
f.close()
