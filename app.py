from SelectAndSearch import *
#  创建主界面
app = tk.Tk()
app.title('基于内容的图像检索（CBIR）')
background = tk.PhotoImage(file="icon/nex.gif")  # 背景图片
background2 = tk.PhotoImage(file="icon/bk2.GIF")  # 背景图片

#  添加背景和标题
bg = tk.Label(app, image=background, compound=tk.CENTER, bg="#989cb8")
bg.place(relx=0, rely=0, relwidth=1, relheight=1)
title = tk.Label(app, text='基于内容的图像检索（CBIR）', font=("宋体", 30), image=background2, compound=tk.CENTER)
title.place(relx=0.2, rely=0.05, relwidth=0.6, relheight=0.15)


SelectAndSearch(app)
app.mainloop()
