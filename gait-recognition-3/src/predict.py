#coding=utf-8
#from __future__ import print_function
import numpy as np
import tensorflow as tf
import imutils
import cv2
import os
import json
from testdb import dbmain
def draw_person(image, persont):
    x, y, w, h = persont
    cv2.rectangle(image, (x, y), (x + w, y + h), (255), 2)
def noise(img):
     #定义结构元素
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2, 2))
            #开闭运算，先开运算去除背景噪声，再继续闭运算填充目标内的孔洞
            opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
            cv2.imshow("closed", closed)
            return closed

def predict(path):

    print("In predicting")
    # import the necessary packages
    X = []
    #model = tf.keras.models.load_model('./model/090.h5')
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    cap = cv2.VideoCapture(path)
    framerate = cap.get(cv2.CAP_PROP_FPS)
    print(framerate)
    #fgbg = cv2.createBackgroundSubtractorMOG2(history=50 , detectShadows = 1)
    fgbg = cv2.createBackgroundSubtractorKNN(history = 50,detectShadows=1)
    #firstFrame = None
    while cap.isOpened():
        ret, frame = cap.read()
        if ret!=1:
            break
        frame=cv2.resize(frame,(400,300), interpolation = cv2.INTER_AREA)
        frame_copy = frame.copy()
        fgmask = fgbg.apply(frame_copy)

        #print(frame)
        if frame is not None:
            print("inframe")

            #frames = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#转为灰度值图

            #cv2.imshow("BGR2GRAY", frame)

            #fgmask = fgbg.apply(fgmask, learningRate=0.5)
            #cv2.imshow("fgmask", fgmask)

            ret, thresh = cv2.threshold(fgmask,244,255,cv2.THRESH_BINARY)#转为二值图
            #thresh = noise(thresh)
            cv2.imshow("thresh", thresh)
                     # 对原始帧进行腐蚀膨胀去噪
            thresh = cv2.dilate(thresh, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), iterations=2)
            th = cv2.erode(thresh, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), iterations=2)

            cv2.imshow("th", th)

            contours, _ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            #cv2.imshow("thresh", thresh)


            cimg = np.zeros_like(th)

            print("contours:",len(contours))
            for i in range(len(contours)):
                #x,y,w,h = cv2.boundingRect(contours[i]) #绘制矩形
                #cimg = cv2.rectangle(cimg,(x,y),(x+w,y+h),(255,255,255),2)
                cv2.drawContours(cimg, contours, i, color=(255,255,255), thickness=-1)
                #cimg = noise(cimg)
                cv2.imshow("drawContours", cimg)

                        #绘制人体
            rects, weights = hog.detectMultiScale(frame)
            print("person:",len(rects))
            for person in rects:#定位人体
                draw_person(frame, person)
                x, y, w, h = person
                font=cv2.FONT_HERSHEY_SIMPLEX

                rectss=(rects).tolist()
                persons=person.tolist()
                print(persons,rectss)
                name=str(rectss.index(persons))
                print(name)
                frame= cv2.putText(frame, name, (x, y), font, 1.2, (255, 255, 255), 2)
                  # 图像，文字内容， 坐标 ，字体，大小，颜色，字体厚度
            cv2.imshow("people detection", frame)


            cimg = cv2.cvtColor(th,cv2.COLOR_GRAY2RGB)
            cimg = cv2.resize(cimg,(240,240))
            #cv2.imshow("resize", cimg)
            cimg = cimg.reshape(-1,240,240,3)

            # classes = model.predict(cimg)
            # result = classes.tolist()
            # result2 = result[0]
            # #del result2[0]
            # index = result2.index(max(result2))
            # namedict={"0":"unkonw","1":"person one","2":"person two","3":"person tree","4":"person four"}
            # name = namedict[str(index)]
            # if max(result2)> 0.8:
            #      print(result2)
            #      print(name)
            #      rects, weights = hog.detectMultiScale(frame_copy)
            #      print("person:",len(rects))
            #      for person in rects:#定位人体
            #          draw_person(frame_copy, person)
            #          font=cv2.FONT_HERSHEY_SIMPLEX
            #          frame_copy = cv2.putText(frame_copy, name, (50, 50), font, 1.2, (255, 255, 255), 2)
            #            # 图像，文字内容， 坐标 ，字体，大小，颜色，字体厚度
            #          cv2.imshow("people detection", frame_copy)
            # else:
            #     print("Unknow")


            print("incvcap")
            # image_name="split"+str(i)+".jpg"
            # cv2.imwrite(os.path.join("videosplit", image_name),cimg)

            #X.append(cimg)
        else:
            break
            print("break")
        k = cv2.waitKey(50) & 0xff
        # if k == ord("q"):
        #     break
    print("Video Read")
    cap.release()
    # print(len(X))
    # prediction = []
    # # for i in X:
    # #     prediction.append(model.predict(i))
    # return prediction
class predictclass:
    def __init__(self):
        self.name=""
    def predictgei(self,gei):
            #读取姓名id
        # f = open('directory/user_info.json', 'r', encoding='utf-8')
        # res=f.read()  #读文件
        # data=json.loads(res)
        #"video/fzg2.mp4"
        #ls = predict("video/0303.mp4")
        #print(ls)
        model = tf.keras.models.load_model('./model/090.h5')
        img=gei
        #img = cv2.imread('1.jpg')
        #cv2.imshow('sub',img)
        #cv2.waitKey(0)
        cv2.destroyAllWindows()
        img = cv2.resize(img,(240,240))
        img = img.reshape(-1,240,240,3)
        classes = model.predict(img)
        result = classes.tolist()
        result2 = result[0]
        del result2[0]
        #del result2[0]
        result3=["",""]
        index = result2.index(max(result2))

        namedict=dbmain("SELECT ID, Name  from person")
        try:
            name = namedict[str(index+1)]
        except:
            name = "Unknowd"
        if max(result2)> 0.3 :
             result3[0]=str(index+1)
             result3[1]=name
             print(result2)
             print(name)
        else:
            result3[0]="0"
            result3[1]="Unknow"
            print(result2)
            print("Unknow")
        self.name= result3
    def get_name(self):
        return self.name
if __name__ == '__main__':
    #predict("video/0501.mp4")
    imgt = cv2.imread("gei/001-nm-001-090-0.jpg")
    cv2.imshow('sub',imgt)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    p=predictclass()
    p.predictgei(imgt)