import pickle
import time
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

def hamming(a, b):
	# 汉明距离
    # compute and return the Hamming distance between the integers
	return bin(int(a) ^ int(b)).count("1")




# 待搜索图片路径
def quicklysearch(imagpath):
	# 加载VP和哈希文件
	tree = pickle.loads(open("Feature Library/vptree.pickle", "rb").read())
	hashes = pickle.loads(open("Feature Library/hashes.pickle", "rb").read())
	print("加载 VP-Tree and hashes...")
	image = cv2.imread(imagpath)

	if(len(image)==None):
		print("加载图片失败 ...")

	cv2.namedWindow("test_search",cv2.WINDOW_NORMAL)
	cv2.resizeWindow("test_search",512,384)
	cv2.imshow("test_search", image)

	# 计算哈希值
	search_Hash = dhash(image)
	search_Hash = convert_hash(search_Hash)

	# 获取结果
	distance = 10
	print("搜索中...")
	start = time.time()
	results = tree.get_all_in_range(search_Hash, distance)
	results = sorted(results)
	# print("results:",results)
	end = time.time()
	print("消耗时间 {} s".format(end - start))
	print(results)
	# 结果处理
	for (d, h) in results:
	# 用哈希值获取数据集中的所有相似图像
		resultPaths = hashes.get(h, [])
		print("检索到 {} 张, 汉明距离: {}, 哈希值: {}".format(len(resultPaths), d, h))
		# 显示结果
		for resultPath in resultPaths:
			result = cv2.imread(resultPath)
			cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
			cv2.resizeWindow("Result", 512, 384)
			cv2.imshow("Result", result)
			cv2.waitKey(0)
