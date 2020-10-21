import sys
import tensorflow as tf
from PyQt5.QtWidgets import QMainWindow,QDialog, QApplication, QFileDialog,QMessageBox
#from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage,QPixmap
from gaitGUI import Ui_MainWindow
from PyQt5.QtCore import pyqtSlot
from threading import Thread
# import inspect
# import ctypes
import cv2
import os
# import subprocess
import time
from splitvideo import VideoToFrames
from gei import gei_mains
from train import train_mains
from geiv import geiv_mains
from predict import predictclass
from dataset import Dataset
from testdb import dbmain
import webbrowser
import numpy as np
import pickle

class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.setupUi(self)
        #第一页参数
        self.video_path=""
        self.train_mode=""
        self.frames=0
        self.noise=[2,2,2]
        self.name=""
        self.gei=1
        self.pkl=0
        #第二页参数
        self.pkl_path=""
        self.train_set=[0,0,0,0,0]
        #第三页参数
        self.video_path_p3=""
        self.train_mode_p3=""
        self.frames_p3=0
        self.noise_p3=[2,2,2]
        self.name_p3=""
        model = tf.keras.models.load_model('./model/090.h5')

    #菜单栏
    @pyqtSlot()
    def on_action1_triggered(self):
         files, ok1 =QFileDialog.getOpenFileNames(self,"视频帧文件夹","./vsil","All Files (*);;Text Files (*.txt)")
    @pyqtSlot()
    def on_action2_triggered(self):
        files, ok1 =QFileDialog.getOpenFileNames(self,"步态能量图文件夹","./gei","All Files (*);;Text Files (*.txt)")
    @pyqtSlot()
    def on_action3_triggered(self):
        gei_mains(False)
    @pyqtSlot()
    def on_action4_triggered(self):
        f=Dataset()
        f.run()
    @pyqtSlot()
    def on_action_triggered(self):
        path=r"{}/SQLiteStudio/SQLiteStudio.exe".format(os.getcwd())
        print(path)
        os.system("{}/SQLiteStudio/SQLiteStudio.exe".format(os.getcwd()))
        # child=childWindow()
        # child.show()
        # child.exec_()
    #界面跳转
    #录入界面第一页
    @pyqtSlot()
    def on_pushButton_13_clicked(self):
        self.stackedWidget.setCurrentIndex(0)
        stop_camre(1)
    @pyqtSlot()
    def on_pushButton_14_clicked(self):
        self.stackedWidget.setCurrentIndex(1)
        stop_camre(1)
    @pyqtSlot()
    def on_pushButton_15_clicked(self):
        self.stackedWidget.setCurrentIndex(2)
        stop_camre(1)
        model = tf.keras.models.load_model('./model/090.h5')
    #获取视频文件路径
    @pyqtSlot()
    def on_pushButton_2_clicked(self):
        self.video_path, ok = QFileDialog.getOpenFileName(self, '打开视频', 'C:/', 'All files (*)')
        self.lineEdit_2.setText(self.video_path)

    def get_video_path(self):
        if self.radioButton_2.isChecked():
            self.video_path=0
        elif self.radioButton_3.isChecked():
            self.video_path=1
        else:
            if self.lineEdit_2.text() :
                self.video_path=self.lineEdit_2.text()
            else:
                QMessageBox.information(self, '信息提示','您未选择视频路径',QMessageBox.Yes | QMessageBox.No,QMessageBox.Yes)
                return -1
        print("input video path is : ",self.video_path)
    #获取背景分割模式参数
    def get_train_mode(self):
        if self.radioButton_4.isChecked():
            self.train_mode= "knn"
        elif self.radioButton_5.isChecked():
            self.train_mode="mog"
        self.frames=self.horizontalSlider.value()
        print("trian mode is : ",self.train_mode)
        print("trian frames is :",self.horizontalSlider.value())
    #获取降噪参数
    def get_noise(self):
        self.noise[0]=self.spinBox_2.value()
        self.noise[1]=self.spinBox_3.value()
        self.noise[2]=self.spinBox.value()
        print("noise set :",self.noise)
    #是否生成Gei和打包
    def get_pkl(self):
        self.gei=self.checkBox_3.isChecked()
        self.pkl=self.checkBox_4.isChecked()
        print("gei and pkl is :",self.gei,self.pkl)
    #视频播放功能
    @pyqtSlot()
    def on_pushButton_7_clicked(self):
        #使用线程,否则程序卡死
        stop_camre(0)
        self.get_video_path()
        vp=self.video_path
        th=Thread(target=show_camre(vp,1))
        th.start()

    @pyqtSlot()
    def on_pushButton_8_clicked(self):
        stop_camre(1)
        print(thstop)


    @pyqtSlot()
    def on_pushButton_9_clicked(self):
        self.on_pushButton_7_clicked()
    def on_pushButton_28_clicked(self):
        save_video(1)
    @pyqtSlot()
    def on_pushButton_10_clicked(self):
        self.on_pushButton_8_clicked()
        save_video(0)
    def get_directory(self):
        if self.lineEdit.text() :
               self.name=self.lineEdit.text()
        else:
               QMessageBox.information(self, '信息提示','您未输入姓名',QMessageBox.Yes | QMessageBox.No,QMessageBox.Yes)
               return -1
        print("input name is : ",self.name)

     #录入按键，检查video和name
    @pyqtSlot()
    def on_pushButton_clicked(self):
        self.pushButton.setEnabled(False)
        #判断是否有视频输入
        if self.get_video_path()==-1:
            self.pushButton.setEnabled(True)
            return -1
        #判断是否有姓名输入
        if self.get_directory()==-1:
            self.pushButton.setEnabled(True)
            return -1
        self.get_train_mode()
        self.get_noise()
        self.get_pkl()

        progress=20
        self.progressBar.show()
        self.progressBar.setProperty("value",progress)
        #视频提取前景并保存为帧
        if self.video_path==0 and 1 :
            self.video_path='temp/camera_test.mp4'

        print(self.video_path,self.name,self.train_mode,self.frames,self.noise)
        f=VideoToFrames()
        f.run(self.video_path,self.name,self.train_mode,self.frames,self.noise)

        progress=60
        self.progressBar.setProperty("value",progress)
        #生成gei并打包，在纯色背景下可用
        if self.gei:
            gei_mains(self.pkl)

        progress=100
        self.progressBar.setProperty("value",progress)
        #恢复按键结束视频线程
        self.pushButton.setEnabled(True)
        stop_camre(1)
        #self.progressBar.hide()
    def SetPic(self,img,page):
        # self.lable.setPixmap(QPixmap(imgPath))
        #图片显示
        #print("insetpic")
        if page==1:
            self.label_12.setPixmap(QPixmap.fromImage(img))
            self.label_12.setScaledContents (True)
        if page==3:
            self.label_14.setPixmap(QPixmap.fromImage(img))
            self.label_14.setScaledContents (True)
        if page==4:
            self.label_19.setPixmap(QPixmap.fromImage(img))
            self.label_19.setScaledContents (True)
        if page==5:
            self.label_16.setPixmap(QPixmap.fromImage(img))
            self.label_16.setScaledContents (0)
        if page==6:
            self.label_18.setPixmap(QPixmap.fromImage(img))
            self.label_18.setScaledContents (0)
    #训练界面第二页
    @pyqtSlot()
    def on_pushButton_25_clicked(self):
        self.stackedWidget.setCurrentIndex(0)
        stop_camre(1)
    @pyqtSlot()
    def on_pushButton_26_clicked(self):
        self.stackedWidget.setCurrentIndex(1)
        stop_camre(1)
    @pyqtSlot()
    def on_pushButton_27_clicked(self):
        self.stackedWidget.setCurrentIndex(2)
        stop_camre(1)
        model = tf.keras.models.load_model('./model/090.h5')
    def get_pkl_path(self):
        if self.lineEdit_3.text() :
            self.pkl_path=self.lineEdit_3.text()
        elif self.radioButton_11.isChecked():
            self.pkl_path="pkl/dictionary.pkl"
        else:
            QMessageBox.information(self, '信息提示','您未选择训练数据路径',QMessageBox.Yes | QMessageBox.No,QMessageBox.Yes)
            return -1
        print("input pkl path is : ",self.pkl_path)
    def detial(self):
        os.system("cd ./tmp/keras_log/ && tensorboard --logdir=./")
    def op_detial(self):
        global page2_thread
        page2_thread=Thread(target=self.detial)
        page2_thread.start()
    def get_train_set(self):
        self.train_set[0]=90
        self.train_set[1]=self.horizontalSlider_3.value()
        self.train_set[2]=self.doubleSpinBox.value()
        self.train_set[3]=self.checkBox.isChecked()
        self.train_set[4]=self.checkBox_2.isChecked()
    @pyqtSlot()
    def on_pushButton_21_clicked(self):
        webbrowser.open("http://localhost:6006")
    @pyqtSlot()
    def on_pushButton_6_clicked(self):
        #判断是否选择了训练数据
        if self.get_pkl_path()==-1:
            return -1
        self.get_train_set()
        self.progressBar_3.show()
        self.progressBar_3.setValue(10)
        print(self.pkl_path,["090"],self.train_set[1],self.train_set[2],self.train_set[3],self.train_set[4])
        train_mains(self.pkl_path,["090"],self.train_set[1],self.train_set[2],self.train_set[3],self.train_set[4])
        self.progressBar_3.setValue(100)
        #展示流程图
        if self.train_set[3]==1:
            pix = QPixmap('results/model_plot.png')
            self.label_11.setPixmap(pix)
        #展示详细数据
        if self.train_set[4]==1:
            self.op_detial()
            time.sleep(3)
            self.widget.setEnabled(1)

    #识别界面第三页
    @pyqtSlot()
    def on_pushButton_18_clicked(self):
        self.stackedWidget.setCurrentIndex(0)
        stop_camre(1)
    @pyqtSlot()
    def on_pushButton_19_clicked(self):
        self.stackedWidget.setCurrentIndex(1)
    @pyqtSlot()
    def on_pushButton_20_clicked(self):
        self.stackedWidget.setCurrentIndex(2)
        stop_camre(1)
        model = tf.keras.models.load_model('./model/090.h5')
        #获取视频文件路径
    @pyqtSlot()
    def on_pushButton_3_clicked(self):
        self.video_path_p3, ok = QFileDialog.getOpenFileName(self, '打开视频', 'C:/', 'All files (*)')
        self.lineEdit_4.setText(self.video_path_p3)
    def get_video_path_p3(self):
        print("ingetpath3")
        if self.radioButton_7.isChecked():
            self.video_path_p3=0
        elif self.radioButton_8.isChecked():
            self.video_path_p3=1
        else:
            if self.lineEdit_4.text() :
                self.video_path_p3=self.lineEdit_4.text()
            else:
                QMessageBox.information(self, '信息提示','您未选择视频路径',QMessageBox.Yes | QMessageBox.No,QMessageBox.Yes)
                return -1
        print("input video path is : ",self.video_path_p3)
    #获取背景分割模式参数
    def get_train_mode_p3(self):
        if self.radioButton_9.isChecked():
            self.train_mode_p3= "knn"
        elif self.radioButton_10.isChecked():
            self.train_mode_p3="mog"
        self.frames_p3=self.horizontalSlider_2.value()
        print("trian mode is : ",self.train_mode_p3)
        print("trian frames is :",self.horizontalSlider_2.value())
    #获取降噪参数
    def get_noise_p3(self):
        self.noise_p3[0]=self.spinBox_5.value()
        self.noise_p3[1]=self.spinBox_6.value()
        self.noise_p3[2]=self.spinBox_4.value()
        print("noise set :",self.noise_p3)
    #视频播放功能
    @pyqtSlot()
    def on_pushButton_16_clicked(self):
        #使用线程,否则程序卡死
        stop_camre(0)
        #self.get_video_path_p3()
        if self.get_video_path_p3()==-1:
            return -1
        #self.get_video_path_p3()
        vp3=self.video_path_p3
        global pre
        pre =True
        th3=Thread(target=predict(vp3,3,self.train_mode_p3,self.frames_p3,self.noise_p3))
        th3.start()

    @pyqtSlot()
    def on_pushButton_17_clicked(self):
        stop_camre(1)
        print(thstop)
    @pyqtSlot()
    def on_pushButton_4_clicked(self):
         #使用线程,否则程序卡死
        stop_camre(0)
        if self.get_video_path_p3()==-1:
            return -1
        self.get_train_mode_p3()
        self.get_noise_p3()
        vp3=self.video_path_p3
        mode=self.train_mode_p3
        frames=self.frames_p3
        noise=self.noise_p3
        global pre
        pre=False
        th3=Thread(target=predict(vp3,3,mode,frames,noise))
        th3.start()



#视频显示及保存模块
def show_camre(cm,p):
        #参数0代表系统第一个摄像头,第二就用1 以此类推
        cap=cv2.VideoCapture(cm)
        #设置显示分辨率和FPS ,不设置的话会非常卡
        #cap.set(cv2.CAP_PROP_FRAME_WIDTH,800)
        #cap.set(cv2.CAP_PROP_FRAME_HEIGHT,600)
        #cap.set (cv2.CAP_PROP_FPS,20)
         #视频保存设置
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        #size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        no=len(os.listdir('temp'))

        out = cv2.VideoWriter('temp/camera{}.mp4'.format(no+1), fourcc,fps, (800,600))
        while cap.isOpened():
            #print("inwhile",svideo)
            if thstop:
                return 0
            ret,frame=cap.read()
            if ret!=False or svideo==1:
                #水平翻转
                #frame=cv2.flip(frame,1)
                frame=cv2.resize(frame,(800,600), interpolation = cv2.INTER_AREA)
                if svideo==1:
                    out.write(frame)
                #opencv 默认图像格式是rgb qimage要使用BRG,这里进行格式转换,不用这个的话,图像就变色了
                frame=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
                #mat-->qimage
                a=QImage(frame.data,frame.shape[1],frame.shape[0],QImage.Format_RGB888)
                ui.SetPic(a,p)
                cv2.waitKey(25)
            else:
                break
        cap.release()
        out.release()



def draw_person(image, persont):
    x, y, w, h = persont
    cv2.rectangle(image, (x, y), (x + w, y + h), (255), 2)

def predict(path,lab,train_mode,frames,noise):

    print("In predicting")
    # import the necessary packages
    #X = []
    #model = tf.keras.models.load_model('./model/090.h5')
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    cap = cv2.VideoCapture(path)
    framerate = cap.get(cv2.CAP_PROP_FPS)
    print(framerate)
    if train_mode=="knn":
        fgbg = cv2.createBackgroundSubtractorKNN(history = frames,detectShadows=1)
    else:
        fgbg = cv2.createBackgroundSubtractorMOG2(history=frames , detectShadows = 1)

    #firstFrame = None
    while cap.isOpened():
        if thstop:
            return
        ret, frame = cap.read()
        if ret!=1:
            return
        frame=cv2.resize(frame,(400,300), interpolation = cv2.INTER_AREA)
        frame_copy = frame.copy()
        fgmask = fgbg.apply(frame_copy)
        global pre
        if pre:
            #opencv 默认图像格式是rgb qimage要使用BRG,这里进行格式转换,不用这个的话,图像就变色了
            frame_copy=cv2.cvtColor(frame_copy,cv2.COLOR_RGB2BGR)
            #mat-->qimage
            frame_copy=QImage(frame_copy.data,frame_copy.shape[1],frame_copy.shape[0],QImage.Format_RGB888)
            ui.SetPic(frame_copy,lab)
            cv2.waitKey(25)
            #print("continue")
            continue

        #print(frame)
        if frame is not None:
            #print("inframe")
            ret, thresh = cv2.threshold(fgmask,244,255,cv2.THRESH_BINARY)#转为二值图
            #thresh = noise(thresh)
            #cv2.imshow("thresh", thresh)
                     # 对原始帧进行腐蚀膨胀去噪
            thresh = cv2.dilate(thresh, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (noise[0], noise[1])), iterations=2)
            th = cv2.erode(thresh, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (noise[2], noise[2])), iterations=2)

            #cv2.imshow("th", th)

            contours, _ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)



            cimg = np.zeros_like(thresh)
            #绘制轮廓
            #print("contours:",len(contours))
            for i in range(len(contours)):
                #x,y,w,h = cv2.boundingRect(contours[i]) #绘制矩形
                #cimg = cv2.rectangle(cimg,(x,y),(x+w,y+h),(255,255,255),2)
                cv2.drawContours(cimg, contours, i, color=(255,255,255), thickness=-1)
                #cimg = noise(cimg)
            #cv2.imshow("drawContours", cimg)

            #绘制人体
            rects, weights = hog.detectMultiScale(frame_copy)
            #print("person:",len(rects))
            for person in rects:#定位人体
                draw_person(frame_copy, person)
                x, y, w, h =person

                font=cv2.FONT_HERSHEY_SIMPLEX
                rectss=(rects).tolist()
                persons=person.tolist()
                #print(persons,rectss)
                id=str(rectss.index(persons)+1)
                extract(cimg,id,y,h,x,w)
                frame_copy= cv2.putText(frame_copy, id, (x, y), font, 1.2, (255, 255, 255), 2)
                #   # 图像，文字内容， 坐标 ，字体，大小，颜色，字体厚度
            #cv2.imshow("people detection",frame_copy)
            #
            # #cimg = cv2.cvtColor(th,cv2.COLOR_GRAY2RGB)
            # cimg = cv2.resize(cimg,(240,240))
            # cv2.imshow("resize", cimg)
            # cimg = cimg.reshape(-1,240,240,3)
            #
            #opencv 默认图像格式是rgb qimage要使用BRG,这里进行格式转换,不用这个的话,图像就变色了
            frame_copy=cv2.cvtColor(frame_copy,cv2.COLOR_RGB2BGR)
            #mat-->qimage
            frame_copy=QImage(frame_copy.data,frame_copy.shape[1],frame_copy.shape[0],QImage.Format_RGB888)
            ui.SetPic(frame_copy,lab)

            #opencv 默认图像格式是rgb qimage要使用BRG,这里进行格式转换,不用这个的话,图像就变色了
            frame4=cv2.cvtColor(cimg,cv2.COLOR_RGB2BGR)
            #mat-->qimage
            frame4=QImage(frame4.data,frame4.shape[1],frame4.shape[0],QImage.Format_RGB888)
            ui.SetPic(frame4,4)



            #print("incvcap")
            # image_name="split"+str(i)+".jpg"
            # cv2.imwrite(os.path.join("videosplit", image_name),cimg)

            #X.append(cimg)
        else:
            break
            print("break")
        cv2.waitKey(25)
        # if k == ord("q"):
        #     break
    print("Video Read")
    cap.release()
#提取行人
dictionary=[]
def extract(frame,i,y,h,x,w):

        image_rgb=frame
        top=y
        height=h
        left=x
        width=w

        image_clip = image_rgb[int(top):(int(top) + int(height)), int(left):(int(left) + int(width))]
        frame=image_clip
        image = cv2.resize(frame,(240,240))
        #cv2.imshow("thresh{}".format(i), image)
        if len(dictionary)>30:#设置提取n帧的步态转为gei
            try:
                geiimg=geiv_mains(dictionary)
                #geiimg = geiimg.astype(np.int)
                geiimg=geiimg.astype('uint8')
                #cv2.imshow("gei",geiimg)
                #opencv 默认图像格式是rgb qimage要使用BRG,这里进行格式转换,不用这个的话,图像就变色了
                geiimg=cv2.cvtColor(geiimg,cv2.COLOR_RGB2BGR)
                #mat-->qimage
                geiimgs=QImage(geiimg.data,geiimg.shape[1],geiimg.shape[0],QImage.Format_RGB888)
                #启用多线程识别
                pre=predictclass()
                th=Thread(target=pre.predictgei(geiimg))
                th.start()
                name=pre.get_name()
                #name=predictgei(geiimg)
                show_result(name)
                ui.SetPic(geiimgs,5)
                dictionary.clear()
            except:
                dictionary.clear()
                print("gei error")
        dictionary.append(image)
def show_result(name):
    ui.label_17.setText( "<=current\n"
"\n"
"\n"
"perdict=>\n"
"\n"
"\n"
"{}".format(name[1]))
    if name[0]=="0":
        return
    geiimg=cv2.imread("gei/{0:03}-nm-001-090-0.jpg".format(int(name[0])))

    #opencv 默认图像格式是rgb qimage要使用BRG,这里进行格式转换,不用这个的话,图像就变色了
    geiimg=cv2.cvtColor(geiimg,cv2.COLOR_RGB2BGR)
    #mat-->qimage
    geiimgs=QImage(geiimg.data,geiimg.shape[1],geiimg.shape[0],QImage.Format_RGB888)
    ui.SetPic(geiimgs,6)
#视频资源释放
def stop_camre(bool):
    global thstop
    thstop=bool
def save_video(bool):
    global svideo
    svideo = bool
#多线程控制
# def _async_raise(tid, exctype):
#     """raises the exception, performs cleanup if needed"""
#     tid = ctypes.c_long(tid)
#     if not inspect.isclass(exctype):
#         exctype = type(exctype)
#     res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
#     if res == 0:
#         raise ValueError("invalid thread id")
#     elif res != 1:
#         # """if it returns a number greater than one, you're in trouble,
#         # and you should call it again with exc=NULL to revert the effect"""
#         ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
#         raise SystemError("PyThreadState_SetAsyncExc failed")
# def stop_thread(thread):
#     _async_raise(thread.ident, SystemExit)



# class test:
#     def prints(self):
#         print("hello")
#子页面
# class childWindow(QDialog,Ui_Dialog):
#     def __init__(self):
#         super(childWindow, self).__init__()
#         self.setupUi(self)
#         la=QHBoxLayout()
#         lb=QVBoxLayout()
#         lc=QHBoxLayout()
#         scroll=QScrollArea()
#         a=QWidget()
#         a.setLayout(lb)
#         #lb.addLayout(lc)
#         for x in range(50):
#             lb.addWidget(QPushButton(str(x)))
#         #for x in range(50):
#             #lc.addWidget(QPushButton(str(x)))
#         scroll.setMinimumSize(400,400)
#         #scrollarea 作为一个组件，可以设置窗口
#         scroll.setWidget(a)
#         la.addWidget(scroll)
#         self.setLayout(la)

if __name__ == '__main__':

    thstop=False
    svideo=False
#上面的这个来控制进程结束

    app = QApplication(sys.argv)

    ui = MyWindow()
    ui.progressBar.hide()
    ui.frame_3.hide()
    ui.progressBar_3.hide()
    ui.show()

    app.exec_()
    #sys.exit(app.exec_())
   #退出的时候,结束进程,否则,关不掉进程
    stop_camre(1)
    # global page2_thread
    # page2_thread.join(3)
    #stop_thread(page2_thread)
    print("done")
