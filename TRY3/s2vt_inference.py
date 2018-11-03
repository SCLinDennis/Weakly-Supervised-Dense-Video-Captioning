#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 09:49:40 2018

@author: DennisLin
"""


import tensorflow as tf
import pandas as pd
import numpy as np
import os
import json
import bleu as bl
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from s2vt import Video_Caption_Generator
from os import listdir

#%%
dim_image = 2048
dim_hidden= 100

n_video_lstm_step = 50
n_caption_lstm_step = 20
schedule_sample_probability = 1
n_frame_step = 50
batch_size = 50
model_path = './models/pyf_e-4/'


ixtoword = pd.Series(np.load(model_path+ './ixtoword_pyf.npy').tolist())

bias_init_vector = np.load(model_path + './bias_init_vector_pyf.npy')

model = Video_Caption_Generator(
            dim_image=dim_image,
            n_words=len(ixtoword),
            dim_hidden=dim_hidden,
            batch_size=batch_size,
            n_lstm_steps=n_frame_step,
            n_video_lstm_step=n_video_lstm_step,
            n_caption_lstm_step=n_caption_lstm_step,
            bias_init_vector=bias_init_vector,schedule_p = 1)
video_tf, video_mask_tf, caption_tf, probs_tf, last_embed_tf = model.build_generator()
sess = tf.InteractiveSession()
print("start to restore")
saver = tf.train.Saver()
saver.restore(sess, "./models/pyf_e-4/model2473-86")
print("restore success")

test_folder_path = '../MSRVTT/testing_data_sequence/'
test = json.load(open('../MSRVTT/testing_label.json','r'))
test_path = listdir(test_folder_path)
test_features = [ (file[:-4],np.load(test_folder_path + file)) for file in test_path]
'''
test_feature_dict = {}
for test_tuple in test_features:
    test_feature_dict[test_tuple[0]] = test_tuple[1]
'''




test_sentences = []
id_list = []
for idx, video_feat in test_features:
    video_feat = video_feat.reshape(1,n_frame_step, dim_image)
#    print(video_feat.shape)
    if video_feat.shape[1] == n_frame_step:
        video_mask = np.ones((video_feat.shape[0], video_feat.shape[1]))

    generated_word_index = sess.run(caption_tf, feed_dict={video_tf:video_feat, video_mask_tf:video_mask})
#    print(generated_word_index)
    generated_words = ixtoword[generated_word_index]
    generated_sentence = ' '.join(generated_words)
    generated_sentence = generated_sentence.replace('<bos> ', '')
    generated_sentence = generated_sentence.replace(' <eos>', '')
    generated_sentence = generated_sentence.replace('<pad> ', '')
    generated_sentence = generated_sentence.replace(' <pad>', '')
    generated_sentence = generated_sentence.replace(' <unk>', '')
#    print (generated_sentence,'\n')
    test_sentences.append(generated_sentence)
    id_list.append(idx)
submit = pd.DataFrame(np.array([id_list,test_sentences]).T)
submit.to_csv("prediction.txt",index = False,  header=False)
result = {}
with open("prediction.txt",'r') as f:
    for line in f:
        line = line.rstrip()
        comma = line.index(',')
        test_id = line[:comma]
        caption = line[comma+1:]
        result[test_id] = caption
'''
#count by average
bleu=[]
for item in test:
    score_per_video = []
    for caption in item['caption']:
        caption = caption.rstrip('.')
        score_per_video.append(bl.BLEU(test_sentences[item['id']],caption, 4))
    bleu.append(sum(score_per_video)/len(score_per_video))
average1 = sum(bleu) / len(bleu)
print("Originally, average bleu score is " + str(average1))
'''
#count by the method described in the paper https://aclanthology.info/pdf/P/P02/P02-1040.pdf
bleu=[]
for item in test:
    score_per_video = []
    captions = [x.rstrip('.') for x in item['caption']]
    score_per_video.append(bl.BLEU(result[item['id']],captions, 4, True))
    bleu.append(score_per_video[0])
average = sum(bleu) / len(bleu)
print("By another method, average bleu score is " + str(average))
    
