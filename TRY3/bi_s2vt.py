import numpy as np
import tensorflow as tf
class Video_Caption_Generator():
    def __init__(self, dim_image, n_words, dim_hidden, batch_size, n_lstm_steps, n_video_lstm_step, n_caption_lstm_step, bias_init_vector=None):
        self.dim_image = dim_image
        self.n_words = n_words
        self.dim_hidden = dim_hidden
        self.batch_size = batch_size
        self.n_lstm_steps = n_lstm_steps
        self.n_video_lstm_step=n_video_lstm_step
        self.n_caption_lstm_step=n_caption_lstm_step

        with tf.device("/cpu:0"):
            self.Wemb = tf.Variable(tf.random_uniform([n_words, dim_hidden], -0.1, 0.1), name='Wemb') # (token_unique, 1000)
        
        self.lstm_fw = tf.nn.rnn_cell.BasicLSTMCell(dim_hidden, state_is_tuple=False) # c_state, m_state are concatenated along the column axis 
        self.lstm_bw = tf.nn.rnn_cell.BasicLSTMCell(dim_hidden, state_is_tuple=False)
        self.lstm2 = tf.nn.rnn_cell.BasicLSTMCell(dim_hidden, state_is_tuple=False)

        self.encode_image_W = tf.Variable( tf.random_uniform([dim_image, dim_hidden], -0.1, 0.1), name='encode_image_W') # (4096, 1000)
        self.encode_image_b = tf.Variable( tf.zeros([dim_hidden]), name='encode_image_b')


        self.embed_word_W = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1,0.1), name='embed_word_W') # (1000, n_words)
        if bias_init_vector is not None:
            self.embed_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_word_b')
        else:
            self.embed_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b')

    def build_model(self):
        video = tf.placeholder(tf.float32, [self.batch_size, self.n_video_lstm_step, self.dim_image]) # (batch, 80, 4096)
        video_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_video_lstm_step])

        caption = tf.placeholder(tf.int32, [self.batch_size, self.n_caption_lstm_step+1]) # enclude <BOS>; store word ID; (batch, max_length)
        caption_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_caption_lstm_step+1]) # (batch_size, max_length+1)

        video_flat = tf.reshape(video, [-1, self.dim_image]) 
        image_emb = tf.nn.xw_plus_b( video_flat, self.encode_image_W, self.encode_image_b ) # (batch_size*n_lstm_steps, dim_hidden)
        image_emb = tf.reshape(image_emb, [self.batch_size, self.n_lstm_steps, self.dim_hidden])

        #state1 = tf.zeros([self.batch_size, self.lstm1.state_size]) # initial state
        state2 = tf.zeros([self.batch_size, self.lstm2.state_size]) # initial state
        padding1 = tf.zeros([self.batch_size, self.n_lstm_steps, self.dim_hidden]) # (batch, 1000)
        padding2 = tf.zeros([self.batch_size, self.dim_hidden])
        probs = []
        loss = 0.0

        ##############################  Encoding Stage ##################################
        
        image_emb = tf.unstack(image_emb, self.n_lstm_steps, 1)
        (output1, _, _) = tf.nn.static_bidirectional_rnn(self.lstm_fw, self.lstm_bw, image_emb, dtype=tf.float32)
        for i in range(0, self.n_video_lstm_step): # n_vedio_lstm_step = 80
            with tf.variable_scope("LSTM2", reuse=(i!=0)):
                output2, state2 = self.lstm2(tf.concat( [padding2, output1[i]] ,1), state2)

        ############################# Decoding Stage ######################################


            

        padding1 = tf.unstack(padding1, self.n_lstm_steps, 1)  
        (output1, _, _) = tf.nn.static_bidirectional_rnn(self.lstm_fw, self.lstm_bw, padding1, dtype=tf.float32)
                
        for i in range(0, self.n_caption_lstm_step): ## Phase 2 => only generate captions
            #if i == 0:
            #    current_embed = tf.zeros([self.batch_size, self.dim_hidden])
            #else:
            with tf.device("/cpu:0"):
                current_embed = tf.nn.embedding_lookup(self.Wemb, caption[:, i])
            with tf.variable_scope("LSTM2", reuse= True):
                output2, state2 = self.lstm2(tf.concat([current_embed, output1[i]], 1), state2)

            labels = tf.expand_dims(caption[:, i+1], 1) # (batch, max_length, 1)
            indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1) # (batch_size, 1)
            concated = tf.concat([indices, labels], 1)
            onehot_labels = tf.sparse_to_dense(concated, tf.stack([self.batch_size, self.n_words]), 1.0, 0.0)

            logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit_words, labels= onehot_labels)
            cross_entropy = cross_entropy * caption_mask[:,i]
            probs.append(logit_words)

            current_loss = tf.reduce_sum(cross_entropy)/self.batch_size
            loss = loss + current_loss
            
        return loss, video, video_mask, caption, caption_mask, probs


    def build_generator(self):

        # batch_size = 1
        video = tf.placeholder(tf.float32, [1, self.n_video_lstm_step, self.dim_image]) # (80, 4096)
        video_mask = tf.placeholder(tf.float32, [1, self.n_video_lstm_step])

        video_flat = tf.reshape(video, [-1, self.dim_image])
        image_emb = tf.nn.xw_plus_b(video_flat, self.encode_image_W, self.encode_image_b)
        image_emb = tf.reshape(image_emb, [1, self.n_video_lstm_step, self.dim_hidden])

        #state1 = tf.zeros([1, self.lstm1.state_size])
        state2 = tf.zeros([1, self.lstm2.state_size])
        padding1 = tf.zeros([1, self.n_video_lstm_step, self.dim_hidden])
        padding2 = tf.zeros([1, self.dim_hidden])
        generated_words = []

        probs = []
        embeds = []


        image_emb = tf.unstack(image_emb, self.n_lstm_steps, 1)
        (output1, _, _) = tf.nn.static_bidirectional_rnn(self.lstm_fw, self.lstm_bw, image_emb, dtype=tf.float32)
        for i in range(0, self.n_video_lstm_step):
            with tf.variable_scope("LSTM2", reuse=(i!=0)):
                output2, state2 = self.lstm2(tf.concat([padding2, output1[i]], 1), state2)

        padding1 = tf.unstack(padding1, self.n_lstm_steps, 1)
        (output1, _, _) = tf.nn.static_bidirectional_rnn(self.lstm_fw, self.lstm_bw, padding1, dtype=tf.float32)
        for i in range(0, self.n_caption_lstm_step):
            if i == 0:
                with tf.device('/cpu:0'):
                    current_embed = tf.nn.embedding_lookup(self.Wemb, tf.ones([1], dtype=tf.int64))
            with tf.variable_scope("LSTM2", reuse=True):
                output2, state2 = self.lstm2(tf.concat([current_embed, output1[i]],1), state2)

            logit_words = tf.nn.xw_plus_b( output2, self.embed_word_W, self.embed_word_b)
            max_prob_index = tf.argmax(logit_words, 1)[0]
            generated_words.append(max_prob_index)
            probs.append(logit_words)

            with tf.device("/cpu:0"):
                current_embed = tf.nn.embedding_lookup(self.Wemb, max_prob_index)
                current_embed = tf.expand_dims(current_embed, 0)

            embeds.append(current_embed)

        return video, video_mask, generated_words, probs, embeds
