import asyncio
import csv
from time import ctime
from ColourMoments import *
import cv2
import joblib
from scipy.cluster.vq import *
from sklearn import preprocessing
from pylab import *
from colorFeature import ColorDescriptor
from GLCM import *
from ShapeHistogram import *
import hashing


# 根据颜色特征搜索相关图像
def searchByColor(search_image):
	# 颜色特征描述符初始化
	cd = ColorDescriptor((8, 12, 3))
	# 加载图片
	im = cv2.imread(search_image)
	# 计算获取被检索图像的颜色特征描述符
	features = list(np.array(cd.describe(im)))
	# 获得检索结果
	results= find_img(features)
	path_list=[]
	# 获取结果的图片地址
	for distance,imageID in results:
		path_list.append(imageID)
	# 返回检索结果
	return path_list

# 根据颜色矩搜索相关图像
def searchByCM(search_image):
	# 颜色特征描述符初始化
	# 加载图片
	# 计算获取被检索图像的颜色特征描述符
	features = list(np.array(color_moments(search_image)))
	# 获得检索结果
	results= find_img2(features)
	path_list=[]
	# 获取结果的图片地址
	for distance,imageID in results:
		path_list.append(imageID)
	# 返回检索结果
	return path_list

# 根据纹理特征搜索相关图像（灰度共生矩阵）
def searchByGLCM(search_image):
	# 颜色特征描述符初始化
	# 加载图片
	im = cv2.imread(search_image)
	# 计算获取被检索图像的颜色特征描述符
	features = list(np.array(getglcm(im)))
	# 获得检索结果
	results= find_img3(features)
	path_list=[]
	# 获取结果的图片地址
	for distance,imageID in results:
		path_list.append(imageID)
	# 返回检索结果
	return path_list


# 根据边缘特征搜索相关图像
def searchbyShape(search_image):
	# 颜色特征描述符初始化
	# 加载图片
	im = cv2.imread(search_image)
	# 计算获取被检索图像的颜色特征描述符
	features = list(np.array(ft(im)))
	# 获得检索结果
	results= find_img4(features)
	path_list=[]
	# 获取结果的图片地址
	for distance,imageID in results:
		path_list.append(imageID)
	# 返回检索结果
	return path_list




def cos_sim(a, b):
	vector_a = np.mat(a)
	vector_b = np.mat(b)
	# num = float(vector_a * vector_b.T)
	# denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
	# sim = num / denom
	sim = np.dot(vector_a, vector_b.T)  # T转置,类似numpy.transpose，矩阵的点积

	return sim




def find_img(queryFeatures, limit = 10):
	# 初始化我们的结果字典
	results = {}

	# 打开索引文件进行读取
	with open("Feature Library/colorhis.csv") as f:
		# 初始化 CSV 阅读器
		reader = csv.reader(f)
		# 遍历索引文件中的每一行
		for row in reader:
			# 解析出图像 ID 和特征，然后计算
			# 计算被检索的图像特征与索引文件中的每行的特征的距离
			features = [float(x) for x in row[1:]]
			d = distance(features, queryFeatures)
			print(d)
			# 将计算结果存储
			results[row[0]] = d
		# 关闭阅读器
		f.close()
	results = sorted([(v, k) for (k, v) in results.items()])
	# 返回结果
	# print(len(results))
	return results[:limit]



def find_img2(queryFeatures, limit = 10):
	# 初始化我们的结果字典
	results = {}

	# 打开索引文件进行读取
	with open("Feature Library/colormoment.csv") as f:
		# 初始化 CSV 阅读器
		reader = csv.reader(f)
		# 遍历索引文件中的每一行
		for row in reader:
			# 解析出图像 ID 和特征，然后计算
			# 计算被检索的图像特征与索引文件中的每行的特征的距离
			features = [float(x) for x in row[1:]]
			d = distance(features, queryFeatures)
			print(d)
			# 将计算结果存储
			results[row[0]] = d

		# 关闭阅读器
		f.close()



	results = sorted([(v, k) for (k, v) in results.items()])

	# 返回结果
	# print(len(results))
	return results[:limit]



def find_img3(queryFeatures, limit = 10):
	# 初始化我们的结果字典
	results = {}
	# 打开索引文件进行读取
	with open("Feature Library/GLCM.csv") as f:
		# 初始化 CSV 阅读器
		reader = csv.reader(f)
		# 遍历索引文件中的每一行
		for row in reader:
			# 解析出图像 ID 和特征，然后计算
			# 计算被检索的图像特征与索引文件中的每行的特征的距离
			features = [float(x) for x in row[1:]]
			d = distance(features, queryFeatures)
			print(d)
			# 将计算结果存储
			results[row[0]] = d
		# 关闭阅读器
		f.close()

	results = sorted([(v, k) for (k, v) in results.items()])

	# 返回结果
	# print(len(results))
	return results[:limit]


def find_img4(queryFeatures, limit = 10):
	# 初始化我们的结果字典
	results = {}
	# 打开索引文件进行读取
	with open("Feature Library/ShapeHis.csv") as f:
		# 初始化 CSV 阅读器
		reader = csv.reader(f)
		# 遍历索引文件中的每一行
		for row in reader:
			# 解析出图像 ID 和特征，然后计算
			# 计算被检索的图像特征与索引文件中的每行的特征的距离
			features = [float(x) for x in row[1:]]
			d = distance(features, queryFeatures)
			print(d)
			# 将计算结果存储
			results[row[0]] = d
		# 关闭阅读器
		f.close()

	results = sorted([(v, k) for (k, v) in results.items()])

	# 返回结果
	# print(len(results))
	return results[:limit]




# 根据哈希特征搜索相关图像
def searchBydhash(search_image):

	im = cv2.imread(search_image)
	# 计算获取被检索图像的颜色特征描述符
	features = hashing.convert_hash(hashing.dhash(im))
	# 获得检索结果
	results= find_dhash(features)
	path_list=[]
	# 获取结果的图片地址
	for distance,imageID in results:
		path_list.append(imageID)
	# 返回检索结果
	return path_list


def find_dhash(queryFeatures, limit = 10):
	# 初始化我们的结果字典
	results = {}

	# 打开索引文件进行读取
	with open("Feature Library/dhash.csv") as f:
		# 初始化 CSV 阅读器
		reader = csv.reader(f)
		# 遍历索引文件中的每一行
		for row in reader:
			# 解析出图像 ID 和特征，然后计算
			# 计算被检索的图像特征与索引文件中的每行的特征的距离
			features = [float(x) for x in row[1:]]
			d = hashing.hamming(features[0], queryFeatures)
			print(d)
			# 将计算结果存储
			results[row[0]] = d

		# 关闭阅读器
		f.close()
	results = sorted([(v, k) for (k, v) in results.items()])
	# 返回结果
	# print(len(results))
	return results[:limit]



# 根据哈希特征搜索相关图像
def searchByphash(search_image):

	im = cv2.imread(search_image)
	# 计算获取被检索图像的颜色特征描述符
	features = hashing.convert_hash(hashing.pHash(im))
	# 获得检索结果
	results= find_phash(features)
	path_list=[]
	# 获取结果的图片地址
	for distance,imageID in results:
		path_list.append(imageID)
	# 返回检索结果
	return path_list


def find_phash(queryFeatures, limit = 10):
	# 初始化我们的结果字典
	results = {}

	# 打开索引文件进行读取
	with open("Feature Library/phash.csv") as f:
		# 初始化 CSV 阅读器
		reader = csv.reader(f)
		# 遍历索引文件中的每一行
		for row in reader:
			# 解析出图像 ID 和特征，然后计算
			# 计算被检索的图像特征与索引文件中的每行的特征的距离
			features = [float(x) for x in row[1:]]
			d = hashing.hamming(features[0], queryFeatures)
			print(d)
			# 将计算结果存储
			results[row[0]] = d

		# 关闭阅读器
		f.close()
	results = sorted([(v, k) for (k, v) in results.items()])
	# 返回结果
	# print(len(results))
	return results[:limit]





# 根据哈希特征搜索相关图像
def searchByahash(search_image):

	im = cv2.imread(search_image)
	# 计算获取被检索图像的颜色特征描述符
	features = hashing.convert_hash(hashing.aHash(im))
	# 获得检索结果
	results= find_ahash(features)
	path_list=[]
	# 获取结果的图片地址
	for distance,imageID in results:
		path_list.append(imageID)
	# 返回检索结果
	return path_list


def find_ahash(queryFeatures, limit = 10):
	# 初始化我们的结果字典
	results = {}

	# 打开索引文件进行读取
	with open("Feature Library/ahash.csv") as f:
		# 初始化 CSV 阅读器
		reader = csv.reader(f)
		# 遍历索引文件中的每一行
		for row in reader:
			# 解析出图像 ID 和特征，然后计算
			# 计算被检索的图像特征与索引文件中的每行的特征的距离
			features = [float(x) for x in row[1:]]
			d = hashing.hamming(features[0], queryFeatures)
			print(d)
			# 将计算结果存储
			results[row[0]] = d
		# 关闭阅读器
		f.close()
	results = sorted([(v, k) for (k, v) in results.items()])
	# 返回结果
	# print(len(results))
	return results[:limit]









def distance(histA, histB, eps = 1e-10):
	# 计算卡方距离
	d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
		for (a, b) in zip(histA, histB)])
	# 返回卡方距离
	return d



# 根据sift特征搜索相关图像
def searchBySift(search_image):
	# 加载分类器、类名、缩放器、簇数和词汇
	im_features, image_paths, idf, numWords, voc = joblib.load("Feature Library/bow.pkl")

	# 创建特征提取和关键点检测器对象
	sift_det = cv2.SIFT_create()

	# 列出所有描述符的存储位置
	des_list = []

	im = cv2.imread(search_image)
	gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
	kp, des = sift_det.detectAndCompute(gray, None)

	des_list.append((search_image, des))

	# 将所有描述符垂直堆叠在一个 numpy 数组中
	descriptors = des_list[0][1]

	test_features = np.zeros((1, numWords), "float32")
	words, distance = vq(descriptors, voc)
	for w in words:
		test_features[0][w] += 1

	# 执行 Tf-Idf 矢量化和 L2 归一化
	test_features = test_features * idf
	test_features = preprocessing.normalize(test_features, norm='l2')


	score = np.dot(test_features, im_features.T)
	rank_ID = np.argsort(-score)

	result_list=[]

	for i, ID in enumerate(rank_ID[0][0:10]):
		result_list.append(image_paths[ID])

	return result_list






