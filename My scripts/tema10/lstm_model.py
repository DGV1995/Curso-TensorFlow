#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 16:37:11 2019

@author: diegogarcia-viana
"""

class LSTM_Model():
    def __init__(self, rnn_size, batch_size, learning_rate, training_seq_length, vocabulary_size, infer=False):
        self.rnn_size = rnn_size
        self.learning_rate = learning_rate
        self.vocabulary_size = vocabulary_size
        self.infer = infer
        
        # Inferir es generar texto
        if infer:
            self.batch_size = 1
            self.training_seq_length = 1
        else:
            self.batch_size = batch_size
            self.training_seq_length = training_seq_length
            
        self.lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units = rnn_size)
        self.initial_state = self.lstm_cell.zero_state(self.batch_size, tf.float32)
        
        self.x_data = tf.placeholder(dtype=tf.int32, shape=[batch_size, self.training_seq_length])
        self.y_output = tf.placeholderr(dtype=tf.int32, shape=[self.batch_size, self.training_seq_length])
        
        with tf.variable_scope("lstm_vars"):
            weights = tf.get_variable(name="wights", 
                                      shape=[self.rnn_size, self.vocabulary_size], 
                                      dtype=tf.float32, 
                                      initializer=tf.random_normal_initializer())
            bias = tf.get_variable(name="bias", 
                                   shape=[self.vocabulary_size], 
                                   dtype=tf.float32, 
                                   initializer=tf.constant_initializer(0.0))
            embedding_matrix = tf.get_variable(name="embedding_mat", 
                                               shape=[self.vocabulary_size, self.rnn_size], 
                                               dtype=tf.float32, 
                                               initializer=tf.random_normal_initializer())
            embedding_output = tf.nn.embedding_lookup(params=embedding_matrix, ids=self.x_data)
            rnn_input = tf.split(embedding_output, num_or_size_splits=self.training_seq_length, axis=1)
            rnn_input_trimmed = [tf.squeeze(x, [1]) for x in rnn_input]
            
        def inferred_loop(prev, count):
            prev_trans = tf.add(tf.matmul(prev, weights), bias)
            prev_symbol = tf.stop_gradient(tf.argmax(prev_trans, axis=1))
            output = tf.nn.embedding_lookup(params=embedding_matrix, ids=prev_symbol)
            
        decoder = tf.contrib.legacy_seq2seq.rnn_decoder
        outputs, last_state = decoder(rnn_input_trimmed, 
                                      self.initial_state, 
                                      self.lstm_cell, loop_function=inferred_loop if infer else None)
        
        output = tf.reshape(tf.concat(1, outputs, [-1, self.rnn_size]))
        self.logit_output = tf.add(tf.matmul(output, weights), bias)
        self.model_output = tf.nn.softmax(logit_output)
        
        loss_function = tf.contrib.legacy_seq2seq.sequence_loss_by_example
        loss = loss_function([self.logit_output], 
                             [tf.reshape(self.y_output, -1)], 
                             [tf.ones([self.batch_size*self.training_seq_length])], 
                             self.vocabulary_size)
        
        self.cost = tf.reduce_sum(loss)/(self.batch_size * self.training_seq_length)
        self.final_state = last_state
        gradients, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tf.trainable_variables()), 4.5)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_step = optimizer.apply_gradients(zip(gradients, tf.trainable_variables()))
        
    def sample(self, session, words=vec2word, vocab=word2vec, num=10, prime_text="thou art"):
        state = session.run(self.lstm_cell.zero_state(1, tf.float32))
        word_list = prime_text.split()
        for word in word_list[:-1]:
            x = np.zeros((1,1))
            x[0,0] = vocab[word]
            feed_dict = {self.x_data: x, self.initial_state: state}
            [state] = session.run([self.final_state], feed_dict=feed_dict)
            
        out_sentence = prime_text
        word = word_list[-1]
        
        for n in range(num):
            x = np.zeros((1,1))
            x[0,0] = vocab[word]
            feed_dict = {self.x_data: x, self.initial_state: state}
            [model_output, state] = session.run([self.model_output, self.final_state], feed_dict=feed_dict)
            sample = np.argmax(model_output[0])
            
            if sample == 0:
                break
                
            word = words[sample]
            out_sentence = out_sentence + " " + word
            
        return out_sentence