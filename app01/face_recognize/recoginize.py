import os
import sys
from keras import backend as K
import cv2
IMAGE_SIZE = 224
#将输入的图像大小统一
def resize_image(image,height = IMAGE_SIZE,width = IMAGE_SIZE):
    top,bottom,left,right = 0,0,0,0
    #获取图像大小
    h,w,_ = image.shape
    #对于长宽不一的，取最大值
    longest_edge = max(h,w)
    #计算较短的边需要加多少像素
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass
    #定义填充颜色
    BLACK = [0,0,0]

    #给图像增加边界，使图片长、宽等长，cv2.BORDER_CONSTANT指定边界颜色由value指定
    constant_image = cv2.copyMakeBorder(image,top,bottom,left,right,cv2.BORDER_CONSTANT,value=BLACK)

    return cv2.resize(constant_image,(height,width))

from tensorflow.python import keras
from tensorflow.python.keras.applications.vgg16 import VGG16,preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array
import tensorflow as tf
import numpy as np
# CNN网络模型类
class Model:
    def __init__(self):
        self.model = VGG16(include_top=False)

    # 建立模型
        self.label_dict = {
            '0': 'lwk',
            '1': 'lcy',
            '2': 'hjy',
            '3': 'tsm',
            '4': 'yx',
            '5': 'yzl',
            '6': 'zdw'
        }
    #MODEL_PATH = 'E:\\pythonProject\\tensorflow_learn\\face_recognize\\model\\transfer_01-0.95.h5'
    def load_model(self, file_path):
        self.model.load_weights(file_path)
    def refine_base_model(self):
        """
        重新定义全连接层
        :return:
        """
        x = self.model.outputs[0]
        # 在后面增加我们的全连接层
        x = keras.layers.GlobalAveragePooling2D()(x)
        # 3、定义新的迁移模型
        x = keras.layers.Dense(1024, activation=tf.nn.relu)(x)
        y_predict = keras.layers.Dense(2, activation=tf.nn.softmax)(x)

        # 定义一个新的模型
        transfer_model = keras.models.Model(inputs=self.model.inputs, outputs=y_predict)
        self.model = transfer_model
        return transfer_model
    # 识别人脸
    def face_predict(self, image, model):
        # 依然是根据后端系统确定维度顺序
        if K.image_dim_ordering() == 'th' and image.shape != (1, 3, IMAGE_SIZE, IMAGE_SIZE):
            image = resize_image(image)  # 尺寸必须与训练集一致都应该是IMAGE_SIZE x IMAGE_SIZE
            image = image.reshape((1, 3, IMAGE_SIZE, IMAGE_SIZE))  # 与模型训练不同，这次只是针对1张图片进行预测
        elif K.image_dim_ordering() == 'tf' and image.shape != (1, IMAGE_SIZE, IMAGE_SIZE, 3):
            image = resize_image(image)
            image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))
        image = preprocess_input(image)
        predictions = self.model.predict(image)
        if max(predictions[0]) >= 0.9:
            res = np.argmax(predictions, axis=1)
            #self.label_dict[str(res[0])]
            # 返回类别预测结果
            return res[0]
        else:
            return -1


def start():
    student_id_list = []
    #加载模型
    model = Model()
    model.refine_base_model()
    model.load_model(file_path='./app01/face_recognize/model/transfer_01-0.95.h5')

    # 框住人脸的矩形边框颜色
    color = (0, 255, 0)

    # 捕获指定摄像头的实时视频流
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)
    cap.set(cv2.CAP_PROP_FPS, 30)
    # 人脸识别分类器本地存储路径
    cascade_path = cv2.data.haarcascades+"haarcascade_frontalface_default.xml"
    # 使用人脸识别分类器，读入分类器
    cascade = cv2.CascadeClassifier(cascade_path)
    time = 0
    # 循环检测识别人脸
    while True:
        ret, frame = cap.read()  # 读取一帧视频
        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        time = time + 1
        if time % 1 == 0:
            faceRects = cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=2, minSize=(38, 38))
            if len(faceRects) > 0:
                for faceRect in faceRects:
                    x, y, w, h = faceRect

                    # 截取脸部图像提交给模型识别这是谁
                    image = frame[y: y + h, x: x + w]  # (改)
                    # cv2.imshow('face',image)
                    # cv2.waitKey(0)
                    # print(fr'image.shape:{image.shape[0]},image.type:{type(image)}')
                    faceID = model.face_predict(image, model=model)
                    if faceID == 0:
                        if 4 not in student_id_list:
                            student_id_list.append(4)
                    if faceID == 1:
                        if 10 not in student_id_list:
                            student_id_list.append(10)
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness=2)
                    # #face_id判断（改）
                    # for i in range(len(os.listdir('./app01/face_recognize/face_image/data/'))):
                    #     #print(i,faceID)
                    #     if i == faceID:
                    #         # 文字提示是谁
                    #         cv2.putText(frame,os.listdir('./app01/face_recognize/face_image/data/')[i],
                    #                     (x + 30, y + 30),  # 坐标
                    #                     cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                    #                     1,  # 字号
                    #                     (255, 0, 255),  # 颜色
                    #                     2)  # 字的线宽
            time = 0
        cv2.imshow("recoginize", frame)
        # 等待10毫秒看是否有按键输入
        k = cv2.waitKey(10)
        # 如果输入q则退出循环
        if k & 0xFF == ord('q'):
            break
        del (ret)
        del (frame)
    # 释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()
    return student_id_list