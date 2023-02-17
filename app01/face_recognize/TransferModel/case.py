import tensorflow
from tensorflow.python import keras
from tensorflow.python.keras.applications.vgg16 import VGG16,preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
class TransferModel(object):
    def __init__(self):
        # 定义测试集和训练集的图像增强方法
        self.train_generator = ImageDataGenerator(rescale=1.0 / 255.0)
        self.test_generator = ImageDataGenerator(rescale=1.0 / 255.0)

        self.train_dir = './app01/face_recognize/TransferModel/data/train/'
        self.test_dir = './app01/face_recognize/TransferModel/data/test/'
        # self.train_dir = './data/train/'
        # self.test_dir = './data/test/'
        self.target_size = (224,224)
        self.batch_size = 32

        # 加载基础的模型
        self.base_model = VGG16(include_top=False)

        self.label_dict = {
            '0': 'lcy',
            '1': 'lwk',
            '2': 'yx',
            '3': 'zdw',
        }

    def get_local_data(self):
        """
        获取本地数据
        :return:
        """
        train_gen = self.train_generator.flow_from_directory(self.train_dir,
                                                            target_size=self.target_size,
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            class_mode='binary'
                                                            )

        test_gen = self.train_generator.flow_from_directory(self.test_dir,
                                                             target_size=self.target_size,
                                                             batch_size=self.batch_size,
                                                             shuffle=True,
                                                             class_mode='binary'
                                                             )

        return train_gen, test_gen

    def refine_base_model(self):
        """
        重新定义全连接层
        :return:
        """
        x = self.base_model.outputs[0]
        # 在后面增加我们的全连接层
        x = keras.layers.GlobalAveragePooling2D()(x)
        # 3、定义新的迁移模型
        x = keras.layers.Dense(1024, activation=tf.nn.relu)(x)
        y_predict = keras.layers.Dense(2, activation=tf.nn.softmax)(x)

        # 定义一个新的模型
        transfer_model = keras.models.Model(inputs=self.base_model.inputs, outputs=y_predict)
        return transfer_model

    def freeze_model(self):
        """
        冻结VGG模型（5blocks）
        冻结VGG的多少，根据你的数据量
        :return:
        """
        # self.base_model.layers 获取所有层，返回层的列表
        for layer in self.base_model.layers:
            layer.trainable = False

    def compile(self, model):
        """
        编译模型
        :return:
        """
        model.compile(optimizer=keras.optimizers.Adam(),
                      loss=keras.losses.sparse_categorical_crossentropy,
                      metrics=['accuracy'])
        return None

    def fit_generator(self, model, train_gen, test_gen):
        """
        训练模型，model.fit_generator()不是选择model.fit()
        :return:
        """
        # 每一次迭代准确率记录的h5文件#/app01/face_recognize/TransferModel
        modelckpt = keras.callbacks.ModelCheckpoint('./app01/face_recognize/TransferModel/ckpt/transfer_{epoch:02d}-{val_acc:.2f}.h5',
                                                     monitor='val_acc',
                                                     save_weights_only=True,
                                                     save_best_only=True,
                                                     mode='auto',
                                                     period=1)

        history = model.fit_generator(train_gen, epochs=2, validation_data=test_gen, callbacks=[modelckpt])
        # acc = history.history['sparse_categorical_accuracy']
        # val_acc = history.history['val_sparse_categorical_accuracy']
        # loss = history.history['loss']
        # val_loss = history.history['val_loss']
        # plt.subplot(1, 2, 1)
        # plt.plot(acc, label='Training Accuracy')
        # plt.plot(val_acc, label='Validation Accuracy')
        # plt.title('Training and Validation Accuracy')
        # plt.legend()
        #
        # plt.subplot(1, 2, 2)
        # plt.plot(loss, label='Training Loss')
        # plt.plot(val_loss, label='Validation Loss')
        # plt.title('Training and Validation Loss')
        # plt.legend()
        # plt.show()

        return None

    def predict(self, model):
        """
        预测类别
        :return:
        """

        # 加载模型，transfer_model
        model.load_weights("./ckpt/transfer_01-1.00.h5")

        # 读取图片，处理
        image1 = load_img("./data/train/lcy/25.jpg", target_size=(224, 224))
        image2 = load_img("./data/train/lcy/27.jpg", target_size=(224, 224))
        image3 = load_img("./data/train/lcy/29.jpg", target_size=(224, 224))
        image4 = load_img("./data/train/lcy/30.jpg", target_size=(224, 224))
        image5 = load_img("./data/train/lcy/33.jpg", target_size=(224, 224))
        image6 = load_img("./data/train/lcy/35.jpg", target_size=(224, 224))
        image7 = load_img("./data/train/lcy/45.jpg", target_size=(224, 224))
        image8 = load_img("./data/train/lcy/46.jpg", target_size=(224, 224))
        image9 = load_img("./data/train/lcy/48.jpg", target_size=(224, 224))
        image10 = load_img("./data/train/lcy/52.jpg", target_size=(224, 224))

        images = [image1,image2,image3,image4,
                  image5,image6,image7,image8,
                  image9,image10]
        for image in images:
            image = img_to_array(image)
            #print(image.shape)
            # 四维(224, 224, 3)---》（1， 224， 224， 3）
            img = image.reshape([1, image.shape[0], image.shape[1], image.shape[2]])
            #print(img)
            # model.predict()

            # 预测结果进行处理
            image = preprocess_input(img)
            predictions = model.predict(image)
            #print(predictions)
            res = np.argmax(predictions, axis=1)
            print(self.label_dict[str(res[0])])


def start():
    tm = TransferModel()
    train_gen,test_gen = tm.get_local_data()
    model = tm.refine_base_model()
    tm.freeze_model()
    tm.compile(model)
    tm.fit_generator(model,train_gen,test_gen)
    #tm.predict(model)

if __name__=="__main__":
    start()


