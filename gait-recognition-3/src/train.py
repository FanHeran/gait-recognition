import pickle
import tensorflow as tf
import keras
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
from keras import Sequential
from keras.layers import Conv2D,MaxPooling2D,Activation,Dropout,Flatten, Reshape,BatchNormalization,Dense
import random
import cv2
import os
from keras import backend as K
from tensorflow.keras.utils import plot_model

def Initialize():
    K.tensorflow_backend._get_available_gpus()


def LoadPickle(path):
    print("Generating Sweet Pickle")
    # with open('X.pkl', 'rb') as f:
    #     X = pickle.load(f)
    #     f.close()
    # with open('Y.pkl', 'rb') as f:
    #     Y = pickle.load(f)
    #     f.close()
    with open(path,'rb') as f:
        dictionary_angles = pickle.load(f)
        f.close()
        count = len(set(dictionary_angles["090_Y"]))+1
    print("Pickle is sweet")
    return dictionary_angles,count

def CreatModel(count):
    model = Sequential()
    model.add(Conv2D(16,(7,7),strides=1,input_shape=(240,240,3)))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.75, epsilon=0.0001))
    model.add(MaxPooling2D(pool_size=(2,2),strides=2))

    model.add(Conv2D(16,(7,7),strides=1))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.75, epsilon=0.0001))
    model.add(MaxPooling2D(pool_size=(2,2),strides=2))

    model.add(Conv2D(16,kernel_size=(7,7),strides=1))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.75, epsilon=0.0001))
    model.add(MaxPooling2D(pool_size=(2,2),strides=2))

    model.add(Conv2D(64, kernel_size=(7,7), strides=1))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(count))
    model.add(Activation("sigmoid"))

    model.compile(loss="binary_crossentropy",
                  optimizer="adam",
                  metrics=['accuracy'])
    return model

def Train(model,dictionary_angles,angles,fraq,prec):
    #angles = ["{0:03}".format(i) for i in range(0, 181, 18)]
    #angles = ["090"]
    log_filepath = 'tmp/keras_log'
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(lr=0.001), metrics=['accuracy'])
    tb_cb = keras.callbacks.TensorBoard(log_dir=log_filepath, write_images=1, histogram_freq=1)
    # 设置log的存储位置，将网络权值以图片格式保持在tensorboard中显示，设置每一个周期计算一次网络的权值，每层输出值的分布直方图
    cbks = [tb_cb]
    for angle in angles:
        X = dictionary_angles[angle + '_X']
        Y = dictionary_angles[angle + '_Y']
        combined = list(zip(X, Y))
        random.shuffle(combined)
        X[:], Y[:] = zip(*combined)
        x_train,  y_train = X, Y
        y_train = to_categorical(y_train)
        x_train = np.asarray(x_train)
        model.fit(x_train, y_train, epochs=fraq, batch_size=18, callbacks=cbks, validation_split=prec)
        model.save("./model/"+angle + '.h5', overwrite=True)

def Evalute(dictionary_angles,angles):
    result_dictionary = {}
    for angle in angles:
        model = tf.keras.models.load_model("./model/"+angle + '.h5')
        for angle2 in angles:
            print(angle, angle2)
            x_test, y_test = dictionary_angles[angle2 + '_X'], dictionary_angles[angle2 + '_Y']
            combined = list(zip(x_test, y_test))
            random.shuffle(combined)
            x_test[:], y_test[:] = zip(*combined)
            y_test = to_categorical(y_test)
            x_test = np.asarray(x_test)
            test_loss, test_acc = model.evaluate(x_test, y_test)
            result_dictionary[angle + '_' + angle2] = test_acc
    # model.fit(x_train, y_train, epochs=10,batch_size=18,validation_split=0.1)
    # model.save('all_angle.h5',overwrite=True)
    # model = tf.keras.models.load_model('all_angle.h5')
    # test_loss, test_acc = model.evaluate(x_test, y_test)
    for result in result_dictionary:
        print(str(result_dictionary[result]))
def PlotModel(model):
    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
    plot_model(model, to_file='./results/model_plot.png', show_shapes=True, show_layer_names=True)

def train_mains(path,angle,fraq,perc,graph,evlu):
    Initialize()
    training = LoadPickle(path)
    model = CreatModel(training[1])
  # angles = ["{0:03}".format(i) for i in range(0, 181, 18)]
    #angles = ["090"]
    Train(model,training[0],angle,fraq,perc)
    if graph==1:
        Evalute(training[0],angle)
    if evlu==1:
        PlotModel(model)

if __name__ == "__main__":
    path="pkl/dictionary.pkl"
    angle=["090"]
    fraq=30
    perc=0.2
    graph=1
    evlu=1
    train_mains(path,angle,fraq,perc,graph,evlu)