import os
import sys
import time
import json
import random
import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from os import listdir
from keras.preprocessing import sequence
from functools import reduce
from s2vt import Video_Caption_Generator
import bleu as bl


os.environ['CUDA_VISIBLE_DEVICES'] = "0"
#%%

dim_image = 2048#feature dimension
n_frame_step = 30#feature sample
n_video_lstm_step = 30#?????
dim_hidden= 256
n_caption_lstm_step = 30 #?????
schedule_sample_probability = 1
n_epochs = 1000
batch_size = 32
learning_rate = 0.001
decay_epoch = 30
sample_size = 6513
train_feature_folder = '../MSRVTT/training_data_sequence/'
test_folder_path = '../MSRVTT/training_data_sequence/'
train_label_path = '../MSRVTT/training_label.json'
test_label_path = '../MSRVTT/training_label.json'
model_path = './models/pyf_e-4'
word_count_threshold = 7
#%%



def preProBuildWordVocab(sentence_iterator, word_count_threshold=5):
    # borrowed this function from NeuralTalk
    print ('preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold))
    word_counts = {}
    nsents = 0
    for sent in sentence_iterator:
        nsents += 1
        for w in sent.lower().split(' '):
            word_counts[w] = word_counts.get(w, 0) + 1
    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]

    print ('filtered words from %d to %d' % (len(word_counts), len(vocab)))

    ixtoword = {}
    ixtoword[0] = '<pad>'
    ixtoword[1] = '<bos>'
    ixtoword[2] = '<eos>'
    ixtoword[3] = '<unk>'

    wordtoix = {}
    wordtoix['<pad>'] = 0
    wordtoix['<bos>'] = 1
    wordtoix['<eos>'] = 2
    wordtoix['<unk>'] = 3

    for idx, w in enumerate(vocab):
        wordtoix[w] = idx+4
        ixtoword[idx+4] = w

    word_counts['<pad>'] = nsents
    word_counts['<bos>'] = nsents
    word_counts['<eos>'] = nsents
    word_counts['<unk>'] = nsents

    bias_init_vector = np.array([1.0 * word_counts[ ixtoword[i] ] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range
    
    return wordtoix, ixtoword, bias_init_vector
#%%
path_list = listdir(train_feature_folder)
train_feature_path = [train_feature_folder+path for path in path_list]
#### train_ID_list ####
train_ID_list = [ path[:-4] for path in path_list]


#### train_feature_dict ####
train_feature_dict = {}
for path in train_feature_path:
    feature = np.load(path)
    train_feature_dict[path[:-4].replace(train_feature_folder,"")] = feature

def clean_string(string):
    return string.replace('.', '').replace(',', '').replace('"', '').replace('\n', '').replace('?', '').replace('!', '').replace('\\', '').replace('/', '')

train_label_dict={}
with open(train_label_path) as data_file:    
    train_label = json.load(data_file)
captions_corpus = []
for sample in train_label:
    cleaned_captions = [clean_string(sentence) for sentence in sample["caption"]]
    captions_corpus += cleaned_captions
    train_label_dict[sample["id"]] = cleaned_captions
    #print(cleaned_captions)


wordtoix, ixtoword, bias_init_vector = preProBuildWordVocab(captions_corpus, word_count_threshold)
np.save("./wordtoix_pyf", wordtoix)
np.save('./ixtoword_pyf', ixtoword)
np.save("./bias_init_vector_pyf", bias_init_vector)


model = Video_Caption_Generator(
            dim_image=dim_image,
            n_words=len(wordtoix),
            dim_hidden=dim_hidden,
            batch_size=batch_size,
            n_lstm_steps=n_frame_step,
            n_video_lstm_step=n_video_lstm_step,
            n_caption_lstm_step=n_caption_lstm_step,
            bias_init_vector=bias_init_vector,
            schedule_p = schedule_sample_probability)

tf_loss, tf_video, tf_video_mask, tf_caption, tf_caption_mask, tf_probs= model.build_model()

## testing graph
video_tf, video_mask_tf, caption_tf, probs_tf, last_embed_tf = model.build_generator()

test_path = listdir(test_folder_path)
test_features = [ (file[:-4],np.load(test_folder_path + file)) for file in test_path]
test = json.load(open(test_label_path,'r'))
ixtoword_series = pd.Series(np.load('./ixtoword_pyf.npy').tolist())


sess = tf.InteractiveSession()
saver = tf.train.Saver(max_to_keep=100)

train_op = tf.train.AdamOptimizer(0.001).minimize(tf_loss)
tf.global_variables_initializer().run()

loss_fd = open('loss3_pyf.txt', 'w')
bleu_fd = open('bleu3_pyf.txt', 'w')
loss_to_draw = []


X_y_pairs = []
target_texts = []

words_list = []
for ID in train_ID_list:
#     text = np.random.choice(train_label_dict[ID],1)[0]
    for text in train_label_dict[ID]:
        X_y_pairs.append((train_feature_dict[ID],text))
        words = text.split()
        target_texts.append(words)
        for word in words:
            words_list.append(word)
target_words_set = np.unique(words_list, return_counts=True)[0]
num_decoder_tokens = len(target_words_set)
max_decoder_seq_length = max([len(txt) for txt in target_texts])
print("sample counts", np.shape(X_y_pairs))
print("number of decoder tokens", num_decoder_tokens)
print("max length of label", max_decoder_seq_length)

print("tuple first element shape", X_y_pairs[1][0].shape)
print("tuple second element:", X_y_pairs[1][1])



print("tuple second element:", X_y_pairs[1][1])
loss_to_draw_epoch = []
print("Total sample:", sample_size)
for epoch in range(0, n_epochs):
    random.shuffle(X_y_pairs)
    X_y_pairs_sub =  random.sample(X_y_pairs,sample_size)
    # modify schedule p
    if epoch<decay_epoch:
        model.schedule_p = 1
    else:
        #linear
        model.schedule_p = np.max([1-(epoch/decay_epoch-1), 0])
        
        #inversesigmoid decay
        #model.schedule_p = decay_epoch/(decay_epoch+np.exp(epoch/decay_epoch))
    
    for batch_start, batch_end in zip(range(0, sample_size, batch_size), range(batch_size, sample_size, batch_size)):
        start_time = time.time()
        current_batch = X_y_pairs_sub[batch_start:batch_end]
        current_feats = [ row[0] for row in current_batch]
        #current_feats = np.array(current_feats)
        #current_feats = np.reshape(current_feats, (32, 1, 11264))
        current_video_masks = np.zeros((batch_size, n_video_lstm_step))

        current_captions = np.array(["<bos> "+ row[1] for row in current_batch])
        for idx, single_caption in enumerate(current_captions):
            word = single_caption.lower().split(" ")
            if len(word) < n_caption_lstm_step:
                current_captions[idx] = current_captions[idx] + " <eos>"
            else:
                new_word = ""
                for i in range(n_caption_lstm_step-1):
                    new_word = new_word + word[i] + " "
                current_captions[idx] = new_word + "<eos>"
        current_caption_ind = []
        for cap in current_captions:
            current_word_ind = []
            for word in cap.lower().split(' '):
                if word in wordtoix:
                    current_word_ind.append(wordtoix[word])
                else:
                    current_word_ind.append(wordtoix['<unk>'])
            current_caption_ind.append(current_word_ind)

        current_caption_matrix = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=n_caption_lstm_step)
        current_caption_matrix = np.hstack( [current_caption_matrix, np.zeros( [len(current_caption_matrix), 1] ) ] ).astype(int)
        current_caption_masks = np.zeros( (current_caption_matrix.shape[0], current_caption_matrix.shape[1]) )
        nonzeros = np.sum((current_caption_matrix != 0),1) # 算每個row 有幾個字
        for ind, row in enumerate(current_caption_masks):
            row[:nonzeros[ind]] = 1 # 把前幾個有字的在mask上塗成1
        '''
        probs_val = sess.run(tf_probs, feed_dict={
                    tf_video:current_feats,
                    tf_caption: current_caption_matrix
                    })
        '''
        _, loss_val = sess.run(
                        [train_op, tf_loss],
                        feed_dict={
                            tf_video: current_feats,
                            tf_video_mask : current_video_masks,
                            tf_caption: current_caption_matrix,
                            tf_caption_mask: current_caption_masks
                            })
        loss_to_draw_epoch.append(loss_val)
    print (" Epoch: ", epoch, " loss: ", loss_val, ' Elapsed time: ', str((time.time() - start_time)))
    loss_fd.write('epoch ' + str(epoch) + ' loss ' + str(loss_val) + '\n')
    
    test_sentences = []
    id_list = []

    #validation
    random.shuffle(test_features)
    test_features_sub =  random.sample(test_features, 50)
    for idx, video_feat in test_features_sub:
        video_feat = video_feat.reshape(1, n_frame_step, dim_image)
        if video_feat.shape[1] == n_frame_step:
            video_mask = np.ones((video_feat.shape[0], video_feat.shape[1]))
        generated_word_index = sess.run(caption_tf, feed_dict={video_tf:video_feat, video_mask_tf:video_mask})
        generated_words = ixtoword_series[generated_word_index]
        generated_sentence = ' '.join(generated_words)
        generated_sentence = generated_sentence.replace('<bos> ', '')
        generated_sentence = generated_sentence.replace(' <eos>', '')
        generated_sentence = generated_sentence.replace('<pad> ', '')
        generated_sentence = generated_sentence.replace(' <pad>', '')
        generated_sentence = generated_sentence.replace(' <unk>', '')
        id_list.append(idx)
        test_sentences.append(generated_sentence)
        if idx in ["video6632","video6551","video6514","video6590","video6587"] and np.mod(epoch, 5) == 0:
            print(idx)
            print(generated_sentence)
    submit = pd.DataFrame(np.array([id_list,test_sentences]).T)
    submit.to_csv("ADL_hw2_pyf.txt",index = False,  header=False)
    result = {}
    with open("ADL_hw2_pyf.txt",'r') as f:
        for line in f:
            line = line.rstrip()
            comma = line.index(',')
            test_id = line[:comma]
            caption = line[comma+1:]
            result[test_id] = caption
    '''
    Calcalate the bleu score
    '''

    #count by the method described in the paper https://aclanthology.info/pdf/P/P02/P02-1040.pdf
    bleu=[]
    for item in test:
        if item['id'] in id_list:
            score_per_video = []
            captions = [x.rstrip('.') for x in item['caption']]
            score_per_video.append(bl.BLEU(result[item['id']], captions, 4, True))
            bleu.append(score_per_video[0])
    average = sum(bleu) / len(bleu)

    print("By another method, average bleu score is " + str(average))
    
    bleu_fd.write('epoch: ' + str(epoch) + ' bleu: ' + str(average) + '\n')
    loss_to_draw.append(np.mean(loss_to_draw_epoch))
    plt_save_dir = "./loss_imgs"
    plt_save_img_name = str(epoch) + '.png'
    plt.plot(range(len(loss_to_draw)), loss_to_draw, color='g')
    plt.grid(True)
    plt.savefig(os.path.join(plt_save_dir, plt_save_img_name))
    '''
    if np.mod(epoch, 10) == 0:
        print ("Epoch ", epoch, " is done. Saving the model ...")
        saver.save(sess, os.path.join(model_path, 'model'+str(average1)[2:6])+'_'+str(average)[2:5], global_step=epoch)
    '''
