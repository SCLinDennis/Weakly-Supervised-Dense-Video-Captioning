#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 11:35:21 2018

@author: DennisLin
"""
import os
import glob
import numpy as np
from tqdm import tqdm
from keras.models import Model
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from Res_video_bag import pretrained_model
from load_data import load_caption

class Extractor():
    def __init__(self, label_length, layer1, layer2, weight = '/home/ubuntu/SCL/Weakly_Supervised_S2VT/Weight_Resnet50/fcn_weights.best.h5'):
        base_model = pretrained_model(
            img_shape = [320, 320, 3], 
            length = label_length           
        )
        
        base_model.load_weights(weight) 

        self.model = Model(
            inputs=base_model.input,
            outputs=[base_model.get_layer(layer1).output, base_model.get_layer(layer2).output]
        )
        self.model.summary()
        
        
          
    def extract(self, image_path):

        img = image.load_img(image_path, target_size=(320, 320))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        # Get the prediction.
        [features_layer1, features_layer2] = self.model.predict(x)
        #features = features[0]
        return [features_layer1[0], features_layer2[0]]

def region_generation(paths, model):
	region_sequence = np.zeros([30, 2048])
	for idx, path in enumerate(paths):
		[feature_map, probability_map] = model.extract(path) #(4, 4, 6261)
		#feature_map = model2.extract(path, 1)
		if not idx:
			last_sequence, i_out, j_out = find_max_sum(probability_map)
			region_sequence[idx, :] = feature_map[i_out, j_out]
		else:
			last_sequence, i_out, j_out = find_max_dot(probability_map, i_out, j_out, last_sequence)
			region_sequence[idx, :] = feature_map[i_out, j_out]
	return region_sequence


def find_max_sum(probability_map):#(4, 4, 6261)
	max_, i_out, j_out = 0, 0, 0
	for i in range(4):
		for j in range(4):
			sum_tmp = sum(probability_map[i, j])
			ls = [sum_tmp, max_]
			if not ls.index(max(ls)):
				max_, i_out, j_out = sum_tmp, i, j
	return probability_map[i_out, j_out], i_out, j_out#(6261,)


def find_max_dot(probability_map, i_pre, j_pre, last_sequence):
	max_, i_out, j_out = 0, 0, 0
	for i in range(i_pre-1, i_pre+1):
		for j in range(j_pre-1, j_pre+1):
			if i >= 0 and i <= 3 and j >= 0 and j <= 3:
				dot_tmp = np.dot(last_sequence, probability_map[i, j])
				ls = [dot_tmp, max_]
				if not ls.index(max(ls)):
					max_, i_out, j_out = dot_tmp, i, j
	return probability_map[i_out, j_out], i_out, j_out


weight_path = './Weight_Resnet50_vasbag/fcn_weights.best.h5'
train_label_path = './MSRVTT/training_label.json'
test_label_path = './MSRVTT/testing_label.json'
frame_path = './frames/'
train_output_path = './MSRVTT/training_data_sequence/'
test_output_path = './MSRVTT/testing_data_sequence/'
word_count_threshold = 7
if __name__ == '__main__':
	train_label, word2ix, _, _ =  load_caption(train_label_path, word_count_threshold)
	test_label, _, _, _ = load_caption(test_label_path, word_count_threshold)
	model1 = Extractor(len(list(word2ix)), 'avg_pool', 'fc7', weight_path)
#	model2 = Extractor(len(list(word2ix)), 'avg_pool', weight_path)
	train_id_list = list(train_label.keys())
	test_id_list = list(test_label.keys())
	'''
        for id_ in tqdm(sorted(train_id_list)):
		img_paths = glob.glob(frame_path + id_ + '-*.jpg')
		region_sequence = region_generation(img_paths, model1)
		video_path = os.path.join(train_output_path, id_+'.npy')

		np.save(video_path, region_sequence)
		print('Now saving ' + id_ + 'features!')
        '''
	for id_ in tqdm(sorted(test_id_list)):
		img_paths = glob.glob(frame_path + id_ + '-*.jpg')
		region_sequence = region_generation(img_paths, model1)
		video_path = os.path.join(test_output_path, id_+'.npy')

		np.save(video_path, region_sequence)
		print('Now saving ' + id_ + 'features!')

