import os
import numpy as np
import random
import tensorflow as tf
import glob
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.layers import Dropout, Input, Lambda, Convolution2D, MaxPooling2D
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, Callback
from keras import backend as K
from load_data import load_caption

img_dim_ordering = 'tf'
K.set_image_dim_ordering(img_dim_ordering)

def weight_variable(shape):
    #initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.get_variable("W", shape,
           initializer=tf.contrib.layers.xavier_initializer())


def bias_variable(shape):
    initial = tf.constant(0, shape=shape, dtype = tf.float32)
    return tf.Variable(initial)

#Lexical FCN model with MIML loss
def pretrained_model(img_shape, length):
    '''
    img_shape = (320, 320, 3)
    length = 7677
    '''
    keras_input = Input(shape=img_shape, name = 'image_input')        
    model_vgg16_conv = VGG16(input_shape=img_shape, weights='imagenet', include_top=False)
    #model_vgg16_conv.summary()
    output_vgg16_conv = model_vgg16_conv(keras_input)#(batch, 10, 10, 512)
    
    x = Convolution2D(4096, kernel_size=(3, 3), padding = "valid", activation = "relu", name = "fc_6", input_shape=model_vgg16_conv.output_shape[1:])(output_vgg16_conv)
    x = Dropout(0.5, name = 'drop6')(x)
    x = Convolution2D(4096, kernel_size=(1, 1), padding = 'valid', activation = "relu", name = "fc7")(x)
    x = Dropout(0.5, name = 'drop7')(x)
    features =  MaxPooling2D((2, 2), strides=(2, 2), name='block6_pool')(x)
    #print(features)
    w_p = weight_variable([4096, length])
    b_p = bias_variable([length])
    def fcn_layer(features):        
        instance_p = tf.tensordot(features, w_p, axes = [[-1], [0]])
        instance_p = instance_p + b_p
        instance_p = tf.layers.batch_normalization(instance_p,training=True)
        instance_p = tf.sigmoid(instance_p)
        instance_p = tf.reshape(instance_p, [-1, 16, length])
        #print(instance_p.get_shape())#[None, 4, 4, 6863]
        #tmp = K.sum(K.log(1-instance_p), axis = 1)
        #pwi = 1. - tf.exp(tmp)
        tmp = tf.reduce_prod(instance_p, axis = 1)
        pwi = 1. - tmp
#        pwi = tf.subtract(1., tf.exp(tmp))
        return pwi    
    pwi = Lambda(fcn_layer, name = "fc8")(features)
    #print(pwi)
    pretrained_model = Model(inputs = keras_input, outputs = pwi)  
    return pretrained_model

#%%
def loss(y_true, y_pred):
    _epsilon = tf.convert_to_tensor(1e-07, dtype = y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
    loss = y_true * (-K.log(y_pred)) + (1 - y_true) * (-K.log(1 - y_pred))
    print(loss)
    loss = K.mean(loss, axis = -1)  
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
            x = preprocess_input(x)
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
class MyCbk(Callback):

    def __init__(self, model):
         self.model_to_save = model

    def on_epoch_end(self, epoch, logs=None):
        self.model_to_save.save('model_at_epoch_%d.h5' % epoch)      
#%%
frame_path = '/home/ubuntu/SCL/Video-Captioning/DATA/frames/'
train_label_path = '/home/ubuntu/SCL/ADLxMLDS2017/hw2/MSRVTT_hw2_data/training_label.json'
#dataset = "MSRVTT"
batch_size = 16
num_train_video = 6513
frame_size = [320, 320, 3]
num_frame = 30
num_epochs = 10
word_count_threshold = 16
init_lr = 1e-3
#label_length = 7677
if __name__ == '__main__':
    train_label, word2ix, ix2word =  load_caption(train_label_path, word_count_threshold)
    generator = train_generator2(train_label, word2ix, batch_size)
    model = pretrained_model(frame_size, len(list(word2ix)))
    for layer in model.layers[:2]:
        layer.trainable = False
    print(model.summary())
    adam = Adam(lr=init_lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=init_lr/num_epochs, amsgrad=False)
    #sgd = SGD(lr=init_lr, momentum=0.0, decay=init_lr/num_epochs, nesterov=False)
    model.compile(loss=loss, optimizer=adam, metrics=["accuracy"])
    filepath = "fcn_weights.best.h5"    
#    checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [MyCbk(model)]
    model.fit_generator(generator, epochs = num_epochs, steps_per_epoch = num_train_video)
    model.save_weights('fcn_weights.h5')    

'''
# training the model
model = pretrained_model(x_train.shape[1:], len(set(y_train)), 'sigmoid')
model.summary()
hist = model.fit(x_train, y_train, epochs=2, validation_data=(x_test, y_test), verbose=1)
'''
