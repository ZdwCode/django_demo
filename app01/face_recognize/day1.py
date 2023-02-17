
import os

import PIL.Image
import cv2
import sys
from PIL import Image
def CatchUsbVideo(window_name, camera_idx):
    """

    :param window_name: 检测人脸的测试
    :param camera_idx:
    :return:
    """
    cv2.namedWindow(window_name)

    # 视频来源，可以来自一段已存好的视频，也可以直接来自USB摄像头
    cap = cv2.VideoCapture(camera_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)
    cap.set(cv2.CAP_PROP_FPS, 30)
    # 告诉OpenCV使用人脸识别分类器
    classfier = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")

    # 识别出人脸后要画的边框的颜色，RGB格式
    color = (0, 255, 0)
    time = 0
    while cap.isOpened():
        ok, frame = cap.read()  # 读取一帧数据
        if not ok:
            break
        time = time + 1
            # 将当前帧转换成灰度图像
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if time % 1 == 0:

            # 人脸检测，1.2和2分别为图片缩放比例和需要检测的有效点数
            faceRects = classfier.detectMultiScale(grey, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))
            if len(faceRects) > 0:  # 大于0则检测到人脸
                for faceRect in faceRects:  # 单独框出每一张人脸
                    x, y, w, h = faceRect
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)
            time = 0
        # # 显示图像
        cv2.imshow(window_name, frame)
        c = cv2.waitKey(1)
        if c & 0xFF == ord('q'):
            break
        del (ok)
        del (frame)
            # 释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()
def get_face():
    ###################################################
    # 相当于公司人力组织一次所有员工人脸信息采集
    ###################################################
    while True:
        print("是否录入员工信息(Yes or No)?")
        if input() == 'Yes':
            # 员工姓名(要输入英文，汉字容易报错)
            new_user_name = input("请输入您的姓名：")

            print("请看摄像头！")

            # 采集员工图像的数量自己设定，越多识别准确度越高，但训练速度贼慢
            window_name = '信息采集'  # 图像窗口
            camera_id = 0  # 相机的ID号
            images_num = 200  # 采集图片数量
            path = './face_image/data/' + new_user_name  # 图像保存位置

            CatchPICFromVideo(window_name, camera_id, images_num, path)
        else:
            break
if __name__ == '__main__':
    if len(sys.argv) != 1:
        print("Usage:%s camera_id\r\n" % (sys.argv[0]))
    else:
        CatchUsbVideo("recoginize_face", './mp4/test.mp4')
def CreateFolder(path):
    """
    判断地址是否存在
    :param path:
    :return:
    """
    #去除首位空格
    del_path_space = path.strip()
    #去除尾部'\'
    del_path_tail = del_path_space.rstrip('\\')
    #判读输入路径是否已存在
    isexists = os.path.exists(del_path_tail)
    if not isexists:
        os.makedirs(del_path_tail)
        return True
    else:
        return False
def CatchPICFromVideo(window_name,camera_idx,catch_pic_num,path_name):
    ################
    # 获取人的脸部信息，并保存到所属文件夹
    ################
    #检查输入路径是否存在——不存在就创建
    CreateFolder(path_name)

    cv2.namedWindow(window_name)

    # 视频来源，可以来自一段已存好的视频，也可以直接来自USB摄像头
    cap = cv2.VideoCapture(camera_idx)

    # 告诉OpenCV使用人脸识别分类器
    classfier = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")

    #识别出人脸后要画的边框的颜色，RGB格式
    color = (0, 255, 0)

    num = 0
    while cap.isOpened():
        ok, frame = cap.read()  # 读取一帧数据
        if not ok:
            break

        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 将当前桢图像转换成灰度图像

        # 人脸检测，1.2和2分别为图片缩放比例和需要检测的有效点数
        faceRects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=2, minSize=(32, 32))
        if len(faceRects) > 0:  # 大于0则检测到人脸
            for faceRect in faceRects:  # 单独框出每一张人脸
                x, y, w, h = faceRect
                #cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)
                if w > 200:
                    # 将当前帧保存为图片
                    img_name = '%s\%d.jpg' % (path_name, num)
                    #image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                    image = grey[y:y+h,x:x+w]           #保存灰度人脸图
                    cv2.imwrite(img_name, image)
                    num += 1
                    if num > (catch_pic_num):  # 如果超过指定最大保存数量退出循环
                        break

                    #画出矩形框的时候稍微比识别的脸大一圈
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)

                    # 显示当前捕捉到了多少人脸图片了，这样站在那里被拍摄时心里有个数，不用两眼一抹黑傻等着
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame, 'num:%d' % (num), (x + 30, y + 30), font, 1, (255, 0, 255), 4)

        # 超过指定最大保存数量结束程序
        if num > (catch_pic_num): break

        # 显示图像
        cv2.imshow(window_name, frame)
        #按键盘‘Q’中断采集
        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            break

    # 释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()
# if __name__ == '__main__':
#     # if len(sys.argv) != 1:
#     #     print("Usage:%s camera_id\r\n" % (sys.argv[0]))
#     # else:
#     #     CatchUsbVideo("识别人脸区域", 0)
#     get_face()

import sys
import numpy as np
import os
import cv2
################################################
#读取待训练的人脸图像，指定图像路径即可
################################################
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
#读取数据
images = []     #数据集
labels = []     #标注集
def read_path(path_name):
    for dir_item in os.listdir(path_name):
        full_path = path_name + '\\' + dir_item
        if os.path.isdir(full_path):
            read_path(full_path)
        else:
            #判断是人脸照片
            if dir_item.endswith('.jpg'):
                image = cv2.imread(full_path)
                image = resize_image(image)

                images.append(image)
                labels.append(path_name)

    return images,labels

#为每一类数据赋予唯一的标签值
def label_id(label,users,user_num):
    for i in range(user_num):
        if label.endswith(users[i]):
            return i

#从指定位置读数据
def load_dataset(path_name):
    users = os.listdir(path_name)
    user_num = len(users)

    images,labels = read_path(path_name)
    images_np = np.array(images)
    #每个图片夹都赋予一个固定唯一的标签
    labels_np = np.array([label_id(label,users,user_num) for label in labels])

    return images_np,labels_np

# if __name__ == '__main__':
#     if len(sys.argv) != 1:
#         print("Usage:%s path_name\r\n" % (sys.argv[0]))
#     else:
#         images,labels = load_dataset('./face_image/data')
#         #print(labels)


########################
# 人脸特征训练
########################

import random

import os
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K

IMAGE_SIZE = 224


class Dataset:
    def __init__(self, path_name):
        # 训练集
        self.train_images = None
        self.train_labels = None

        # 验证集
        self.valid_images = None
        self.valid_labels = None

        # 测试集
        self.test_images = None
        self.test_labels = None

        # 数据集加载路径
        self.path_name = path_name
        # 图像种类
        self.user_num = len(os.listdir(path_name))
        # 当前库采用的维度顺序
        self.input_shape = None

    # 加载数据集并按照交叉验证的原则划分数据集并进行相关预处理工作
    def load(self, img_rows=IMAGE_SIZE, img_cols=IMAGE_SIZE,
             img_channels=3):
        # 数据种类
        nb_classes = self.user_num
        # 加载数据集到内存
        images, labels = load_dataset(self.path_name)

        train_images, valid_images, train_labels, valid_labels = train_test_split(images, labels, test_size=0.3,
                                                                                  random_state=random.randint(0, 100))
        # _, test_images, _, test_labels = train_test_split(images, labels, test_size=0.5,
        #                                                   random_state=random.randint(0, 100))

        # 当前的维度顺序如果为'th'，则输入图片数据时的顺序为：channels,rows,cols，否则:rows,cols,channels
        # 这部分代码就是根据keras库要求的维度顺序重组训练数据集
        if K.image_dim_ordering() == 'th':
            train_images = train_images.reshape(train_images.shape[0], img_channels, img_rows, img_cols)
            valid_images = valid_images.reshape(valid_images.shape[0], img_channels, img_rows, img_cols)
            # test_images = test_images.reshape(test_images.shape[0], img_channels, img_rows, img_cols)
            self.input_shape = (img_channels, img_rows, img_cols)
        else:
            train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, img_channels)
            valid_images = valid_images.reshape(valid_images.shape[0], img_rows, img_cols, img_channels)
            # test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, img_channels)
            self.input_shape = (img_rows, img_cols, img_channels)

            # 输出训练集、验证集、测试集的数量
            print(train_images.shape[0], 'train samples')
            print(valid_images.shape[0], 'valid samples')
            # print(test_images.shape[0], 'test samples')

            # 我们的模型使用categorical_crossentropy作为损失函数，因此需要根据类别数量nb_classes将
            # 类别标签进行one-hot编码使其向量化，在这里我们的类别只有两种，经过转化后标签数据变为二维
            train_labels = np_utils.to_categorical(train_labels, nb_classes)
            valid_labels = np_utils.to_categorical(valid_labels, nb_classes)
            # test_labels = np_utils.to_categorical(test_labels, nb_classes)

            # 像素数据浮点化以便归一化
            train_images = train_images.astype('float32')
            valid_images = valid_images.astype('float32')
            # test_images = test_images.astype('float32')

            # 将其归一化,图像的各像素值归一化到0~1区间
            train_images /= 255
            valid_images /= 255
            # test_images /= 255

            self.train_images = train_images
            self.valid_images = valid_images
            # self.test_images = test_images
            self.train_labels = train_labels
            self.valid_labels = valid_labels
            # self.test_labels = test_labels

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
            '0': 'hjy',
            '1': 'lk',
            '2': 'lwk',
            '3': 'zh',
        }
    def build_model(self, dataset, nb_classes=4):

        # 构建一个空的网络模型，它是一个线性堆叠模型，各神经网络层会被顺序添加，专业名称为序贯模型或线性堆叠模型
        self.model = Sequential()

        # 以下代码将顺序添加CNN网络需要的各层，一个add就是一个网络层
        self.model.add(Convolution2D(32, 3, 3, border_mode='same',
                                     input_shape=dataset.input_shape))  # 1 2维卷积层
        self.model.add(Activation('relu'))  # 2 激活函数层

        self.model.add(Convolution2D(32, 3, 3))  # 3 2维卷积层
        self.model.add(Activation('relu'))  # 4 激活函数层

        self.model.add(MaxPooling2D(pool_size=(2, 2)))  # 5 池化层
        self.model.add(Dropout(0.25))  # 6 Dropout层

        self.model.add(Convolution2D(64, 3, 3, border_mode='same'))  # 7  2维卷积层
        self.model.add(Activation('relu'))  # 8  激活函数层

        self.model.add(Convolution2D(64, 3, 3))  # 9  2维卷积层
        self.model.add(Activation('relu'))  # 10 激活函数层

        self.model.add(MaxPooling2D(pool_size=(2, 2)))  # 11 池化层
        self.model.add(Dropout(0.25))  # 12 Dropout层

        self.model.add(Flatten())  # 13 Flatten层
        self.model.add(Dense(512))  # 14 Dense层,又被称作全连接层
        self.model.add(Activation('relu'))  # 15 激活函数层
        self.model.add(Dropout(0.5))  # 16 Dropout层
        self.model.add(Dense(nb_classes))  # 17 Dense层
        self.model.add(Activation('softmax'))  # 18 分类层，输出最终结果

        # 输出模型概况
        self.model.summary()

    # 训练模型
    def train(self, dataset, batch_size=20, nb_epoch=10, data_augmentation=True):
        sgd = SGD(lr=0.01, decay=1e-6,
                  momentum=0.9, nesterov=True)  # 采用SGD+momentum的优化器进行训练，首先生成一个优化器对象
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=sgd,
                           metrics=['accuracy'])  # 完成实际的模型配置工作

        # 不使用数据提升，所谓的提升就是从我们提供的训练数据中利用旋转、翻转、加噪声等方法创造新的
        # 训练数据，有意识的提升训练数据规模，增加模型训练量
        if not data_augmentation:
            self.model.fit(dataset.train_images,
                           dataset.train_labels,
                           batch_size=batch_size,
                           nb_epoch=nb_epoch,
                           validation_data=(dataset.valid_images, dataset.valid_labels),
                           shuffle=True)
        # 使用实时数据提升
        else:
            # 定义数据生成器用于数据提升，其返回一个生成器对象datagen，datagen每被调用一
            # 次其生成一组数据（顺序生成），节省内存，其实就是python的数据生成器
            datagen = ImageDataGenerator(
                featurewise_center=False,  # 是否使输入数据去中心化（均值为0），
                samplewise_center=False,  # 是否使输入数据的每个样本均值为0
                featurewise_std_normalization=False,  # 是否数据标准化（输入数据除以数据集的标准差）
                samplewise_std_normalization=False,  # 是否将每个样本数据除以自身的标准差
                zca_whitening=False,  # 是否对输入数据施以ZCA白化
                rotation_range=20,  # 数据提升时图片随机转动的角度(范围为0～180)
                width_shift_range=0.2,  # 数据提升时图片水平偏移的幅度（单位为图片宽度的占比，0~1之间的浮点数）
                height_shift_range=0.2,  # 同上，只不过这里是垂直
                horizontal_flip=True,  # 是否进行随机水平翻转
                vertical_flip=False)  # 是否进行随机垂直翻转

            # 计算整个训练样本集的数量以用于特征值归一化、ZCA白化等处理
            datagen.fit(dataset.train_images)

            # 利用生成器开始训练模型
            self.model.fit_generator(datagen.flow(dataset.train_images, dataset.train_labels,
                                                  batch_size=batch_size),
                                     samples_per_epoch=dataset.train_images.shape[0],
                                     nb_epoch=nb_epoch,
                                     validation_data=(dataset.valid_images, dataset.valid_labels))

    MODEL_PATH = 'E:\\pythonProject\\tensorflow_learn\\face_recognize\\model\\transfer_01-0.82.h5'

    def save_model(self, file_path=MODEL_PATH):
        self.model.save(file_path)

    def load_model(self, file_path=MODEL_PATH):
        self.model.load_weights(file_path)

    def evaluate(self, dataset):
        score = self.model.evaluate(dataset.test_images, dataset.test_labels, verbose=1)
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))
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
        y_predict = keras.layers.Dense(4, activation=tf.nn.softmax)(x)

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
        # 浮点并归一化
        # image = image.astype('float32')
        # image /= 255

        # # 给出输入属于各个类别的概率
        # result_probability = self.model.predict_proba(image)
        # print(f'result_probability{result_probability}')
        # # print('result:', result_probability)
        #
        # # 给出类别预测(改）
        # if max(result_probability[0]) >= 0.7:
        #     result = self.model.predict_classes(image)
        #     print('result:', result)
        #     # 返回类别预测结果
        #     return result[0]
        # else:
        #     return -1

        #re_image = img_to_array(image)
        #img = image.reshape([1, image.shape[0], image.shape[1], image.shape[2]])
        image = preprocess_input(image)
        predictions = self.model.predict(image)
        print(predictions)
        if max(predictions[0]) >= 0.7:
            res = np.argmax(predictions, axis=1)
            #self.label_dict[str(res[0])]
            # 返回类别预测结果
            return res[0]
        else:
            return -1
# if __name__ == '__main__':
#     user_num = len(os.listdir('./face_image/data/'))
#
#     dataset = Dataset('./face_image/data/')
#     dataset.load()
#
#     model = Model()
#     model.build_model(dataset, nb_classes=user_num)
#
#     # 先前添加的测试build_model()函数的代码
#     model.build_model(dataset, nb_classes=user_num)
#     # 测试训练函数的代码
#     model.train(dataset)
#
#     model.save_model(file_path='./model/aggregate.face.model.h5')


import cv2
import sys
import os

# if __name__ == '__main__':
#     if len(sys.argv) != 1:
#         print("Usage:%s camera_id\r\n" % (sys.argv[0]))
#         sys.exit(0)
#
#     #加载模型
#     model = Model()
#     model.refine_base_model()
#     model.load_model(file_path='./model/transfer_01-0.82.h5')
#
#     # 框住人脸的矩形边框颜色
#     color = (0, 255, 0)
#
#     # 捕获指定摄像头的实时视频流
#     cap = cv2.VideoCapture(0)
#
#     # 人脸识别分类器本地存储路径
#     cascade_path = cv2.data.haarcascades+"haarcascade_frontalface_default.xml"
#
#     # 循环检测识别人脸
#     while True:
#         ret, frame = cap.read()  # 读取一帧视频
#
#         if ret is True:
#
#             # 图像灰化，降低计算复杂度
#             frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         else:
#             continue
#         # 使用人脸识别分类器，读入分类器
#         cascade = cv2.CascadeClassifier(cascade_path)
#
#         # 利用分类器识别出哪个区域为人脸
#         faceRects = cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=2, minSize=(38, 38))
#         if len(faceRects) > 0:
#             print(f'识别到了{len(faceRects)}张人脸')
#             for faceRect in faceRects:
#                 x, y, w, h = faceRect
#
#                 # 截取脸部图像提交给模型识别这是谁
#                 image = frame[y: y + h, x: x + w]       #(改)
#                 # cv2.imshow('face',image)
#                 # cv2.waitKey(0)
#                 print(fr'image.shape:{image.shape[0]},image.type:{type(image)}')
#                 faceID = model.face_predict(image,model=model)
#                 print(f'识别到的faceId:{faceID}')
#                 cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness=2)
#                 #face_id判断（改）
#                 print('开始匹配')
#                 for i in range(len(os.listdir('./face_image/data/'))):
#                     print(i,faceID)
#                     if i == faceID:
#                         # 文字提示是谁
#                         cv2.putText(frame,os.listdir('./face_image/data/')[i],
#                                     (x + 30, y + 30),  # 坐标
#                                     cv2.FONT_HERSHEY_SIMPLEX,  # 字体
#                                     1,  # 字号
#                                     (255, 0, 255),  # 颜色
#                                     2)  # 字的线宽
#
#         cv2.imshow("login", frame)
#
#         # 等待10毫秒看是否有按键输入
#         k = cv2.waitKey(10)
#         # 如果输入q则退出循环
#         if k & 0xFF == ord('q'):
#             break
#
#     # 释放摄像头并销毁所有窗口
#     cap.release()
#     cv2.destroyAllWindows()
# #     # window_name = 'test'
# #     # camera_idx = './mp4/temp_sample.MP4'
# #     # #camera_idx = 0
# #     # CatchUsbVideo(window_name, camera_idx)

