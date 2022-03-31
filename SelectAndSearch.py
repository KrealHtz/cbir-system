
import time
from PIL import Image
from PIL import ImageTk
import tkinter as tk
from tkinter import filedialog
from searchbyvgg import *
from searchbyt import *
from getFeatures import *
from searchbyhash import *
import os


class SelectAndSearch(object):
    def __init__(self, master=None):
        self.app = master
        self.app.geometry("1280x720")
        # 启动后创建组件
        self.create()

    def create(self):
        # 创建一个输入框
        img_path = tk.Entry(self.app, font=("宋体", 18), )
        # 顺序布局
        img_path.pack()
        # 坐标
        img_path.place(relx=0.15, rely=0.2, relwidth=0.6, relheight=0.1)

        # 参数是：要适应的窗口宽、高、Image.open后的图片
        # 调整尺寸
        def img_resize(w_box, h_box, pil_image):
            print(pil_image)
            # 获取图像的原始大小
            width, height = pil_image.size
            f1 = 1.0 * w_box / width
            f2 = 1.0 * h_box / height
            factor = min([f1, f2])
            width = int(width * factor)
            height = int(height * factor)
            # 更改图片尺寸，Image.ANTIALIAS：高质量
            return pil_image.resize((width, height), Image.ANTIALIAS)



        #  图标
        img_s = Image.open('icon/search.png')
        img_s_resized = img_resize(0.05 * 1280, 0.1 * 720, img_s)
        self.img_s = ImageTk.PhotoImage(img_s_resized)
        lbs1 = tk.Label(self.app, imag=self.img_s, compound=tk.CENTER, bg='white')
        lbs1.place(relx=0.15, rely=0.2, relwidth=0.05, relheight=0.1)

        #  添加图片图标
        img_t = Image.open('icon/picture.png')
        img_t_resized = img_resize(0.05 * 1280, 0.1 * 720, img_t)
        self.img_t = ImageTk.PhotoImage(img_t_resized)
        lbt = tk.Label(self.app, imag=self.img_t, compound=tk.CENTER, bg='#6888a8')
        lbt.place(relx=0.05, rely=0.05, relwidth=0.05, relheight=0.1)

        #  本地上传图标
        upload = tk.Button(self.app, text="选择图片", font=("宋体", 20), command=lambda: img_choose(img_path))
        upload.place(relx=0.7, rely=0.3, relwidth=0.15, relheight=0.1)

        #  选择数据库
        upload = tk.Button(self.app, text="选择数据库", font=("宋体", 20), command=lambda: db_choose(img_path))
        upload.place(relx=0.7, rely=0.2, relwidth=0.15, relheight=0.1)

        # set
        enter_setdb = tk.Button(self.app, text="提取特征", font=("宋体", 20), command=lambda: getFeature(img_path))
        enter_setdb.place(relx=0.85, rely=0.2, relwidth=0.1, relheight=0.1)


        # 快速检索
        enter_setdb = tk.Button(self.app, text="快速检索", font=("宋体", 20), command=lambda: bang(img_path))
        enter_setdb.place(relx=0.85, rely=0.3, relwidth=0.1, relheight=0.1)

        #  基于VGG检索
        enter_vgg = tk.Button(self.app, text="VGG-16", font=("宋体", 20), command=lambda: enter(3))
        enter_vgg.place(relx=0.7, rely=0.85, relwidth=0.2, relheight=0.1)

        #  基于SIFT特征检索
        enter_sift = tk.Button(self.app, text="SIFT特征检索", font=("宋体", 20), command=lambda: enter(0))
        enter_sift.place(relx=0.425, rely=0.65, relwidth=0.2, relheight=0.1)

        #  基于颜色特征检索
        enter_color = tk.Button(self.app, text="颜色直方图", font=("宋体", 20), command=lambda: enter(1))
        enter_color.place(relx=0.7, rely=0.45, relwidth=0.2, relheight=0.1)

        #  基于HSV检索
        enter_hsv = tk.Button(self.app, text="HSV 中心距法", font=("宋体", 18), command=lambda: enter(2))
        enter_hsv.place(relx=0.425, rely=0.45, relwidth=0.2, relheight=0.1)

        #  基于纹理特征检索
        enter_wl = tk.Button(self.app, text="纹理特征", font=("宋体", 20),command=lambda: enter(4))
        enter_wl.place(relx=0.7, rely=0.65, relwidth=0.2, relheight=0.1)

        #  基于边缘特征检索
        enter_by = tk.Button(self.app, text="边缘特征", font=("宋体", 20), command=lambda: enter(5))
        enter_by.place(relx=0.425, rely=0.85, relwidth=0.2, relheight=0.1)

        #  基于dhash检索
        enter_dhash = tk.Button(self.app, text="dhash", font=("宋体", 20), command=lambda: enter(6))
        enter_dhash.place(relx=0.15, rely=0.85, relwidth=0.2, relheight=0.1)

        #  基于phash检索
        enter_phash = tk.Button(self.app, text="phash", font=("宋体", 20), command=lambda: enter(7))
        enter_phash.place(relx=0.15, rely=0.65, relwidth=0.2, relheight=0.1)

        #  基于ahash检索
        enter_phash = tk.Button(self.app, text="ahash", font=("宋体", 20), command=lambda: enter(8))
        enter_phash.place(relx=0.15, rely=0.45, relwidth=0.2, relheight=0.1)

        #  选择图片
        def img_choose(img_path):
            # 打开文件管理器，选择图片
            self.app.picture = filedialog.askopenfilename(parent=self.app, initialdir=os.getcwd(), title="本地上传")
            # 同时将图片路径写入行内
            # img_path.delete(0,"end")
            img_path.insert(0, self.app.picture)
            # img_path[0] = self.app.picture

        # 选择数据库
        def db_choose(db_path):
            # 打开文件管理器，选择图片
            self.app.db = filedialog.askdirectory(parent=self.app, initialdir=os.getcwd(), title="本地上传")
            # 同时将图片路径写入行内
            # img_path.delete(0,"end")
            db_path.insert(0, self.app.db)
            # img_path[0] = self.app.picture


        def getFeature(img_path):
            search_db = img_path.get()
            print(search_db)
            allget = getAllPics(search_db)
            mainget(allget)
            img_path.delete(0, 200)



        def bang(imag_path):
            search_path = img_path.get()
            quicklysearch(search_path)
            img_path.delete(0, 200)

        def enter(option):
            # 被检索的图像路径
            search_path = img_path.get()
            # 存储检索结果
            im_ls=[]
            # 未选择图片，则不检索
            if (search_path == '基于内容的图像检索（CBIR）'):
                return
            # 计算检索的耗时
            # 获取当前系统时间
            start = time.perf_counter()
            # 0代表使用sift特征进行搜索
            if (option == 0):
                print(search_path)
                im_ls =searchBySift(search_path)
            # 1代表使用颜色特征进行搜索
            elif (option == 1):
                print(search_path)
                im_ls=searchByColor(search_path)
            elif (option == 2):
                print(search_path)
                im_ls=searchByCM(search_path)
            elif (option == 3):
                print(search_path)
                im_ls = searchByVgg(search_path)
            elif (option == 4):
                print(search_path)
                im_ls = searchByGLCM(search_path)
            elif (option == 5):
                print(search_path)
                im_ls = searchbyShape(search_path)

            elif (option == 6):
                print(search_path)
                im_ls = searchBydhash(search_path)

            elif (option == 7):
                print(search_path)
                im_ls = searchByphash(search_path)

            elif (option == 8):
                print(search_path)
                im_ls = searchByahash(search_path)
            # 获取当前系统时间
            end = time.perf_counter()
            # 计算得到检索所用的总时间
            run_time = end - start
            print('运行时长:', run_time)

            # # 获取相似度
            # score = getScores()

            #  关闭主页面，创建结果界面
            self.app.destroy()
            result = tk.Tk()
            result.geometry("1280x720")
            result.title('查询结果')
            photo = tk.PhotoImage(file="icon/nex.gif")  # 背景图片

            background = tk.Label(result, image=photo, compound=tk.CENTER)
            background.place(relx=0, rely=0, relwidth=1, relheight=1)

            backbutton = tk.Button(result, text="返回", font=("宋体", 25), command=lambda: back(result))
            backbutton.place(relx=0.8, rely=0.1, relwidth=0.08, relheight=0.08)

            word1 = tk.Label(result, text='被检索的图片：', font=("宋体", 25), image=photo, compound=tk.CENTER)
            word1.place(relx=0.1, rely=0, relwidth=0.3, relheight=0.07)

            word2 = tk.Label(result, text='检索结果：', font=("宋体", 20), image=photo, compound=tk.CENTER)
            word2.place(relx=0.15, rely=0.4, relwidth=0.18, relheight=0.07)

            #  上传的图片
            img0 = Image.open(search_path)
            img0_resized = img_resize(0.3 * 1280, 0.3 * 720, img0)
            img0 = ImageTk.PhotoImage(img0_resized)
            lb0 = tk.Label(result, imag=img0, compound=tk.CENTER)
            lb0.place(relx=0.1, rely=0.1, relwidth=0.3, relheight=0.3)

            #  十张检索结果图
            img1 = Image.open(im_ls[0])
            img1_resized = img_resize(0.19 * 1280, 0.2 * 720, img1)
            img1 = ImageTk.PhotoImage(img1_resized)
            lb1 = tk.Label(result, imag=img1, compound=tk.CENTER)
            lb1.place(relx=0, rely=0.5, relwidth=0.19, relheight=0.2)

            img2 = Image.open(im_ls[1])
            img2_resized = img_resize(0.19 * 1280, 0.2 * 720, img2)
            img2 = ImageTk.PhotoImage(img2_resized)
            lb2 = tk.Label(result, imag=img2, compound=tk.CENTER)
            lb2.place(relx=0.2, rely=0.5, relwidth=0.19, relheight=0.2)

            img3 = Image.open(im_ls[2])
            img3_resized = img_resize(0.19 * 1280, 0.2 * 720, img3)
            img3 = ImageTk.PhotoImage(img3_resized)
            lb3 = tk.Label(result, imag=img3, compound=tk.CENTER)
            lb3.place(relx=0.4, rely=0.5, relwidth=0.19, relheight=0.2)

            img4 = Image.open(im_ls[3])
            img4_resized = img_resize(0.19 * 1280, 0.2 * 720, img4)
            img4 = ImageTk.PhotoImage(img4_resized)
            lb4 = tk.Label(result, imag=img4, compound=tk.CENTER)
            lb4.place(relx=0.6, rely=0.5, relwidth=0.19, relheight=0.2)

            img5 = Image.open(im_ls[4])
            img5_resized = img_resize(0.19 * 1280, 0.2 * 720, img5)
            img5 = ImageTk.PhotoImage(img5_resized)
            lb5 = tk.Label(result, imag=img5, compound=tk.CENTER)
            lb5.place(relx=0.8, rely=0.5, relwidth=0.19, relheight=0.2)

            img6 = Image.open(im_ls[5])
            img6_resized = img_resize(0.19 * 1280, 0.2 * 720, img6)
            img6 = ImageTk.PhotoImage(img6_resized)
            lb6 = tk.Label(result, imag=img6, compound=tk.CENTER)
            lb6.place(relx=0, rely=0.75, relwidth=0.19, relheight=0.2)

            img7 = Image.open(im_ls[6])
            img7_resized = img_resize(0.19 * 1280, 0.2 * 720, img7)
            img7 = ImageTk.PhotoImage(img7_resized)
            lb7 = tk.Label(result, imag=img7, compound=tk.CENTER)
            lb7.place(relx=0.2, rely=0.75, relwidth=0.19, relheight=0.2)

            img8 = Image.open(im_ls[7])
            img8_resized = img_resize(0.19 * 1280, 0.2 * 720, img8)
            img8 = ImageTk.PhotoImage(img8_resized)
            lb8 = tk.Label(result, imag=img8, compound=tk.CENTER)
            lb8.place(relx=0.4, rely=0.75, relwidth=0.19, relheight=0.2)

            img9 = Image.open(im_ls[8])
            img9_resized = img_resize(0.19 * 1280, 0.2 * 720, img9)
            img9 = ImageTk.PhotoImage(img9_resized)
            lb9 = tk.Label(result, imag=img9, compound=tk.CENTER)
            lb9.place(relx=0.6, rely=0.75, relwidth=0.19, relheight=0.2)

            img10 = Image.open(im_ls[9])
            img10_resized = img_resize(0.19 * 1280, 0.2 * 720, img10)
            img10 = ImageTk.PhotoImage(img10_resized)
            lb10 = tk.Label(result, imag=img10, compound=tk.CENTER)
            lb10.place(relx=0.8, rely=0.75, relwidth=0.19, relheight=0.2)
            result.mainloop()

        #  返回按键
        def back(result):
            # 摧毁当前结果页面
            result.destroy()
            #  创建主界面
            app = tk.Tk()
            app.title('基于内容的图像检索（CBIR）')
            background = tk.PhotoImage(file="icon/nex.gif")  # 背景图片
            background2 = tk.PhotoImage(file="icon/bk2.GIF")  # 背景图片

            #  添加背景和标题
            bg = tk.Label(app, image=background, compound=tk.CENTER)
            bg.place(relx=0, rely=0, relwidth=1, relheight=1)
            title = tk.Label(app, text='基于内容的图像检索（CBIR）', font=("宋体", 30), image=background2, compound=tk.CENTER)
            title.place(relx=0.2, rely=0.05, relwidth=0.6, relheight=0.15)
            SelectAndSearch(app)
            app.mainloop()
