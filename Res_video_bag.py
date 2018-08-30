import os
import numpy as np
import random
import tensorflow as tf
import glob
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.applications.resnet50  import ResNet50
from keras.applications.resnet50  import preprocess_input as r_preprocess
from keras.layers import Dropout, Input, Lambda, Convolution2D, MaxPooling2D, AveragePooling2D, Flatten, Dense
from keras.engine.topology import Layer
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, Callback
from keras import backend as K
from keras import regularizers
from load_data import load_caption

img_dim_ordering = 'tf'
K.set_image_dim_ordering(img_dim_ordering)
tf.reset_default_graph()

#Lexical FCN model with MIML loss
def pretrained_model(img_shape, length):
    '''
    img_shape = (320, 320, 3)
    length = 7677
    '''
    keras_input = Input(shape=img_shape, name = 'image_input')        
    model_resnet50_conv = ResNet50(input_shape=img_shape, weights='imagenet', include_top=False)
    #model_vgg16_conv.summary()
    x = model_resnet50_conv(keras_input)#(batch, 10, 10, 512)
    print(model_resnet50_conv.summary())
    x = AveragePooling2D(pool_size = (3, 3), strides = (3, 3), padding = 'same', name='avg_pool')(x)
    x = Dense(length, activation = 'sigmoid', use_bias = True, kernel_initializer='glorot_normal',  bias_initializer='zeros', input_shape = (None, 4, 4, 2048), name='fc7')(x)
    features = MaxPooling2D(pool_size=(4, 4), name = 'pool_7')(x)
    def reduce(features):
        features_2D = K.reshape(features, [-1, length])
        return features_2D
    features_2D = Lambda(reduce)(features)
    pretrained_model = Model(inputs = keras_input, outputs = features_2D)  
    return pretrained_model

#%%
def loss(y_true, y_pred):
    _epsilon = tf.convert_to_tensor(1e-07, dtype = y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
    
    tmp = K.sum(K.log(1-y_pred), axis = 0)
    y_pwi = 1. - K.exp(tmp)
    y_true = K.mean(y_true, axis = 0) 
    loss = y_true * (-K.log(y_pwi)) + (1 - y_true) * (-K.log(1 - y_pwi))
    return loss


def train_generator(train_label_dict, word2ix, batch_size):
    label_length = len(word2ix.keys())
    id_list = list(train_label_dict.keys())
    batch_x = np.zeros([batch_size, 320, 320, 3])
    batch_y = np.zeros([batch_size, label_length])
    while True:
        batch_id = random.sample(id_list, batch_size)
        for ix, id_ in enumerate(batch_id): #u'video5397'
            #insert batch_x
            img_paths = glob.glob(frame_path+id_ + '-' + '*.jpg')
            img_path = random.choice(img_paths)
            img = image.load_img(img_path, target_size=(320, 320))
            x = image.img_to_array(img)
            x = preprocess_input(x)
            batch_x[ix] = x#(320, 320, 3)
            
            #insert batch_y
            sentences = train_label_dict[id_]
            y = label_preprocess(sentences, word2ix)            
            batch_y[ix] = y
        yield batch_x, batch_y


def train_generator2(train_label_dict, word2ix, batch_size):
    label_length = len(word2ix.keys())
    id_list = list(train_label_dict.keys())
    batch_x = np.zeros([batch_size, 320, 320, 3])
    batch_y = np.zeros([batch_size, label_length])
    while True:
        id_ = random.choice(id_list)
        img_paths = glob.glob(frame_path+id_ + '-' + '*.jpg')
        img_path = random.sample(img_paths, batch_size)

        sentences = train_label_dict[id_]
        for idx, path in enumerate(img_path):
            img = image.load_img(path, target_size=(320, 320))
            x = image.img_to_array(img)
            x = r_preprocess(x)
            batch_x[idx] = x#(320, 320, 3)

            #insert y
            batch_y[idx] = label_preprocess(sentences, word2ix)

        yield batch_x, batch_y

        
def label_preprocess(sentences, wordtoix):
    label = [0]*len(wordtoix.keys())
    for sentence in sentences:
        words = sentence.split()
        for word in words:
            if word in wordtoix:
                label[wordtoix[word]] = 1    
    return label
#%%
frame_path = './frames/'
train_label_path = './MSRVTT/training_label.json'
dic_path = './dic/'
#dataset = "MSRVTT"
batch_size = 16
num_train_video = 6513
frame_size = [320, 320, 3]
num_frame = 30
num_epochs = 3
word_count_threshold = 7
init_lr = 1e-3
#label_length = 7677
if __name__ == '__main__':
    train_label, word2ix, ix2word, word_counts =  load_caption(train_label_path, word_count_threshold)
    np.save(dic_path+'word2ix.npy', word2ix)
    np.save(dic_path+'ix2word.npy', ix2word)
    np.save(dic_path+'word_counts.npy', word_counts)
    generator = train_generator2(train_label, word2ix, batch_size)
    model = pretrained_model(frame_size, len(list(word2ix)))
    for layer in model.layers[:2]:
        layer.trainable = False
    print(model.summary())
    adam = Adam(lr=init_lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=init_lr/num_epochs, amsgrad=False)
    #sgd = SGD(lr=init_lr, momentum=0.0, decay=init_lr/num_epochs, nesterov=False)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=["accuracy"])
    filepath = "fcn_weights.best.h5"    
    checkpointer = ModelCheckpoint(filepath=filepath, verbose=0, monitor='loss',
                                   save_best_only=True, save_weights_only= True)
    callbacks_list = [checkpointer]
    model.fit_generator(generator, epochs = num_epochs, steps_per_epoch = num_train_video, callbacks=callbacks_list)  

