import tensorflow as tf
import numpy as np
import keras.backend as K
from keras import activations, initializers, regularizers
from keras.engine.topology import Layer
from keras.layers import SimpleRNN
class TensorAttention(Layer):
    '''
    Attention layer that operates on tensors
    '''
    input_ndim = 4
    def __init__(self, att_input_shape, context='word', init='glorot_uniform', activation='tanh', weights=None, hard_k=0, proj_dim = None, rec_hid_dim = None, return_attention=False, **kwargs):
        self.supports_masking = True 
        self.init = initializers.get(init)
        self.activation = activations.get(activation)
        self.context = context
        self.td1, self.td2, self.wd = att_input_shape # (c,w,d)
        self.return_attention = return_attention
        if proj_dim is not None:
            self.proj_dim = proj_dim
        else:
            self.proj_dim = int(self.wd/2) # p
        if rec_hid_dim is not None:
            self.rec_hid_dim = rec_hid_dim
        else:
            self.rec_hid_dim = int(self.proj_dim/2)
        self.initial_weights = weights
        self.hard = True if hard_k>0 else False
        self.k = hard_k
        super(TensorAttention, self).__init__(**kwargs)

    def build(self,input_shape):
        self.att_proj = self.add_weight(name='att_proj',shape=(self.wd, self.proj_dim),
                                        initializer=self.init, trainable=True) # P, (d,p)
        if self.context == 'word':
            self.att_scorer = self.add_weight(name='att_scorer',shape=(self.proj_dim,),initializer=self.init, trainable=True)
        elif self.context == 'clause':
            self.att_scorer = self.add_weight(name='att_scorer',shape=(self.rec_hid_dim,),initializer=self.init, trainable=True)
            self.recurrent_weight = self.add_weight(name='recurrent_weight', 
                                                    shape=(self.rec_hid_dim,self.rec_hid_dim),
                                                    initializer=self.init, trainable=True)
            self.encoder_weight = self.add_weight(name='encoder_weight', 
                                                    shape=(self.proj_dim,self.rec_hid_dim),
                                                    initializer=self.init, trainable=True)
        elif self.context == 'bidirectional_clause':
            self.att_scorer = self.add_weight(name='att_scorer',shape=(self.rec_hid_dim,),initializer=self.init, trainable=True)
            self.recurrent_weight_forward = self.add_weight(name='recurrent_weight_forward', 
                                                    shape=(self.rec_hid_dim,self.rec_hid_dim),
                                                    initializer=self.init, trainable=True)
            self.encoder_weight_forward = self.add_weight(name='encoder_weight_forward', 
                                                    shape=(self.proj_dim,self.rec_hid_dim),
                                                    initializer=self.init, trainable=True)
            self.recurrent_weight_backward = self.add_weight(name='recurrent_weight_backward', 
                                                    shape=(self.rec_hid_dim,self.rec_hid_dim),
                                                    initializer=self.init, trainable=True)
            self.encoder_weight_backward = self.add_weight(name='encoder_weight_backward', 
                                                    shape=(self.proj_dim,self.rec_hid_dim),
                                                    initializer=self.init, trainable=True)

        elif self.context == 'LSTM_clause':
            self.att_scorer = self.add_weight(name='att_scorer',shape=(self.rec_hid_dim,),initializer=self.init, trainable=True)
            self.kernel = self.add_weight(shape=(self.proj_dim, self.rec_hid_dim * 4),
                                name='kernel',initializer=self.init,trainable=True)
            self.recurrent_kernel = self.add_weight(shape=(self.rec_hid_dim, self.rec_hid_dim * 4),
                                name='recurrent_kernel',
                                initializer=self.init,trainable=True)
            self.bias = self.add_weight(shape=(self.rec_hid_dim * 4,),name='bias',
                                        initializer=self.init,trainable=True)
            self.kernel_i = self.kernel[:, :self.rec_hid_dim]
            self.kernel_f = self.kernel[:, self.rec_hid_dim: self.rec_hid_dim * 2]
            self.kernel_c = self.kernel[:, self.rec_hid_dim * 2: self.rec_hid_dim * 3]
            self.kernel_o = self.kernel[:, self.rec_hid_dim * 3:]

            self.recurrent_kernel_i = self.recurrent_kernel[:, :self.rec_hid_dim]
            self.recurrent_kernel_f = (
                self.recurrent_kernel[:, self.rec_hid_dim: self.rec_hid_dim * 2])
            self.recurrent_kernel_c = (
                self.recurrent_kernel[:, self.rec_hid_dim * 2: self.rec_hid_dim * 3])
            self.recurrent_kernel_o = self.recurrent_kernel[:, self.rec_hid_dim * 3:]

            self.bias_i = self.bias[:self.rec_hid_dim]
            self.bias_f = self.bias[self.rec_hid_dim: self.rec_hid_dim * 2]
            self.bias_c = self.bias[self.rec_hid_dim * 2: self.rec_hid_dim * 3]
            self.bias_o = self.bias[self.rec_hid_dim * 3:]

        elif self.context == 'biLSTM_clause':
            self.att_scorer = self.add_weight(name='att_scorer',shape=(self.rec_hid_dim*2,),initializer=self.init, trainable=True)
            self.kernel_forward = self.add_weight(shape=(self.proj_dim, self.rec_hid_dim * 4),
                                name='kernel_forward',initializer=self.init,trainable=True)
            self.recurrent_kernel_forward = self.add_weight(shape=(self.rec_hid_dim, self.rec_hid_dim * 4),
                                name='recurrent_kernel_forward',
                                initializer=self.init,trainable=True)
            self.bias_forward = self.add_weight(shape=(self.rec_hid_dim * 4,),name='bias_forward',
                                        initializer=self.init,trainable=True)
            self.kernel_i_forward = self.kernel_forward[:, :self.rec_hid_dim]
            self.kernel_f_forward = self.kernel_forward[:, self.rec_hid_dim: self.rec_hid_dim * 2]
            self.kernel_c_forward = self.kernel_forward[:, self.rec_hid_dim * 2: self.rec_hid_dim * 3]
            self.kernel_o_forward = self.kernel_forward[:, self.rec_hid_dim * 3:]

            self.recurrent_kernel_i_forward = self.recurrent_kernel_forward[:, :self.rec_hid_dim]
            self.recurrent_kernel_f_forward = (
                self.recurrent_kernel_forward[:, self.rec_hid_dim: self.rec_hid_dim * 2])
            self.recurrent_kernel_c_forward = (
                self.recurrent_kernel_forward[:, self.rec_hid_dim * 2: self.rec_hid_dim * 3])
            self.recurrent_kernel_o_forward = self.recurrent_kernel_forward[:, self.rec_hid_dim * 3:]

            self.bias_i_forward = self.bias_forward[:self.rec_hid_dim]
            self.bias_f_forward = self.bias_forward[self.rec_hid_dim: self.rec_hid_dim * 2]
            self.bias_c_forward = self.bias_forward[self.rec_hid_dim * 2: self.rec_hid_dim * 3]
            self.bias_o_forward = self.bias_forward[self.rec_hid_dim * 3:]

            self.kernel_backward = self.add_weight(shape=(self.proj_dim, self.rec_hid_dim * 4),
                                name='kernel_backward',initializer=self.init,trainable=True)
            self.recurrent_kernel_backward = self.add_weight(shape=(self.rec_hid_dim, self.rec_hid_dim * 4),
                                name='recurrent_kernel_backward',
                                initializer=self.init,trainable=True)
            self.bias_backward = self.add_weight(shape=(self.rec_hid_dim * 4,),name='bias_backward',
                                        initializer=self.init,trainable=True)
            self.kernel_i_backward = self.kernel_backward[:, :self.rec_hid_dim]
            self.kernel_f_backward = self.kernel_backward[:, self.rec_hid_dim: self.rec_hid_dim * 2]
            self.kernel_c_backward = self.kernel_backward[:, self.rec_hid_dim * 2: self.rec_hid_dim * 3]
            self.kernel_o_backward = self.kernel_backward[:, self.rec_hid_dim * 3:]

            self.recurrent_kernel_i_backward = self.recurrent_kernel_backward[:, :self.rec_hid_dim]
            self.recurrent_kernel_f_backward = (
                self.recurrent_kernel_backward[:, self.rec_hid_dim: self.rec_hid_dim * 2])
            self.recurrent_kernel_c_backward = (
                self.recurrent_kernel_backward[:, self.rec_hid_dim * 2: self.rec_hid_dim * 3])
            self.recurrent_kernel_o_backward = self.recurrent_kernel_backward[:, self.rec_hid_dim * 3:]

            self.bias_i_backward = self.bias_backward[:self.rec_hid_dim]
            self.bias_f_backward = self.bias_backward[self.rec_hid_dim: self.rec_hid_dim * 2]
            self.bias_c_backward = self.bias_backward[self.rec_hid_dim * 2: self.rec_hid_dim * 3]
            self.bias_o_backward = self.bias_backward[self.rec_hid_dim * 3:]

        elif self.context == 'para':
            self.att_scorer = self.add_weight(name='att_scorer',shape=(self.td1, self.td2, self.proj_dim),
                                              initializer=self.init, trainable=True) # (c,w,p)
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        super(TensorAttention, self).build(input_shape)
        
    def compute_output_shape(self, input_shape):
        if self.return_attention:
            return [(input_shape[0], input_shape[1], input_shape[3]),(input_shape[0], input_shape[1], input_shape[2])]
        else:  
            return (input_shape[0], input_shape[1], input_shape[3])

    def compute_mask(self, input, input_mask=None):
        if input_mask is not None:
            if self.return_attention:
                return [input_mask[:,:,-1], input_mask]
            else:
                return input_mask[:,:,-1]
        else:
            return None

    def call(self, X, mask=None):
        # input: D (sample,c,w,d)
        proj_input = self.activation(tf.tensordot(X, self.att_proj, axes=[[3],[0]])) # tanh(dot(D,P))=Dl,（sample,c,w,p）
        if self.context == 'word':
            raw_att_scores = tf.tensordot(proj_input, self.att_scorer, axes=[[3],[0]]) # (sample,c,w)
        elif self.context == 'clause':
            def step(X, states):
                new_state = activations.tanh(tf.tensordot(X,self.encoder_weight, axes=[[2],[0]]) \
                    + tf.tensordot(states[0],self.recurrent_weight, axes=[[2],[0]]))
                return new_state, [new_state]
            # Make all-zero initial state. 
            # Directly obtaining the first input dimension is not allowed, so this is the work-aronud.
            initial_state = tf.tensordot(K.max(proj_input*0,axis=2),K.zeros((self.proj_dim, self.rec_hid_dim)), axes = [[2],[0]])
            proj_input_permute = K.permute_dimensions(proj_input,(0,2,1,3))
            _,all_rnn_out,_ = K.rnn(step,proj_input_permute,[initial_state])
            raw_att_scores = tf.tensordot(K.permute_dimensions(all_rnn_out,(0,2,1,3)), 
                                                self.att_scorer, axes=[[3],[0]])
        
        elif self.context == 'bidirectional_clause':
            def step_forward(X, states):
                new_state = activations.tanh(tf.tensordot(X,self.encoder_weight_forward, axes=[[2],[0]]) \
                    + tf.tensordot(states[0],self.recurrent_weight_forward, axes=[[2],[0]]))
                return new_state, [new_state]
            def step_backward(X, states):
                new_state = activations.tanh(tf.tensordot(X,self.encoder_weight_backward, axes=[[2],[0]]) \
                    + tf.tensordot(states[0],self.recurrent_weight_backward, axes=[[2],[0]]))
                return new_state, [new_state]
            # Make all-zero initial state. 
            # Directly obtaining the first input dimension is not allowed, so this is the work-aronud.
            initial_state = tf.tensordot(K.max(proj_input*0,axis=2),K.zeros((self.proj_dim, self.rec_hid_dim)), axes = [[2],[0]]) 
            proj_input_permute = K.permute_dimensions(proj_input,(0,2,1,3))
            proj_input_permute_backward = K.reverse(proj_input_permute, 1)
            _,all_rnn_out_forward,_ = K.rnn(step_forward,proj_input_permute,[initial_state])
            _,all_rnn_out_backward,_ = K.rnn(step_backward,proj_input_permute,[initial_state])
            all_rnn_out = all_rnn_out_forward+all_rnn_out_backward
            raw_att_scores = tf.tensordot(K.permute_dimensions(all_rnn_out,(0,2,1,3)), 
                                                self.att_scorer, axes=[[3],[0]])

        elif self.context == 'LSTM_clause':
            def step(inputs, states):
                h_tm1 = states[0]  # previous memory state
                c_tm1 = states[1]  # previous carry state

                x_i = tf.tensordot(inputs, self.kernel_i,axes=[[2],[0]])
                x_f = tf.tensordot(inputs, self.kernel_f,axes=[[2],[0]])
                x_c = tf.tensordot(inputs, self.kernel_c,axes=[[2],[0]])
                x_o = tf.tensordot(inputs, self.kernel_o,axes=[[2],[0]])
                x_i = K.bias_add(x_i, self.bias_i)
                x_f = K.bias_add(x_f, self.bias_f)
                x_c = K.bias_add(x_c, self.bias_c)
                x_o = K.bias_add(x_o, self.bias_o)
                i = activations.hard_sigmoid(x_i + tf.tensordot(h_tm1,
                                                          self.recurrent_kernel_i,axes=[[2],[0]]))
                f = activations.hard_sigmoid(x_f + tf.tensordot(h_tm1,
                                                          self.recurrent_kernel_f,axes=[[2],[0]]))
                c = f * c_tm1 + i * activations.tanh(x_c + tf.tensordot(h_tm1,
                                                                self.recurrent_kernel_c,axes=[[2],[0]]))
                o = activations.hard_sigmoid(x_o + tf.tensordot(h_tm1,
                                                          self.recurrent_kernel_o,axes=[[2],[0]]))
                h = o * activations.tanh(c)

                return h, [h, c]
            # Make all-zero initial state. 
            # Directly obtaining the first input dimension is not allowed, so this is the work-aronud.
            initial_state = tf.tensordot(K.max(proj_input*0,axis=2),K.zeros((self.proj_dim, self.rec_hid_dim)), axes = [[2],[0]])
            proj_input_permute = K.permute_dimensions(proj_input,(0,2,1,3))
            _,all_rnn_out,_ = K.rnn(step,proj_input_permute,[initial_state,initial_state])
            raw_att_scores = tf.tensordot(K.permute_dimensions(all_rnn_out,(0,2,1,3)), 
                                                self.att_scorer, axes=[[3],[0]])
        elif self.context == 'biLSTM_clause':
            def step_forward(inputs, states):
                h_tm1 = states[0]  # previous memory state
                c_tm1 = states[1]  # previous carry state

                x_i = tf.tensordot(inputs, self.kernel_i_forward,axes=[[2],[0]])
                x_f = tf.tensordot(inputs, self.kernel_f_forward,axes=[[2],[0]])
                x_c = tf.tensordot(inputs, self.kernel_c_forward,axes=[[2],[0]])
                x_o = tf.tensordot(inputs, self.kernel_o_forward,axes=[[2],[0]])
                x_i = K.bias_add(x_i, self.bias_i_forward)
                x_f = K.bias_add(x_f, self.bias_f_forward)
                x_c = K.bias_add(x_c, self.bias_c_forward)
                x_o = K.bias_add(x_o, self.bias_o_forward)
                i = activations.hard_sigmoid(x_i + tf.tensordot(h_tm1,
                                                          self.recurrent_kernel_i_forward,axes=[[2],[0]]))
                f = activations.hard_sigmoid(x_f + tf.tensordot(h_tm1,
                                                          self.recurrent_kernel_f_forward,axes=[[2],[0]]))
                c = f * c_tm1 + i * activations.tanh(x_c + tf.tensordot(h_tm1,
                                                                self.recurrent_kernel_c_forward,axes=[[2],[0]]))
                o = activations.hard_sigmoid(x_o + tf.tensordot(h_tm1,
                                                          self.recurrent_kernel_o_forward,axes=[[2],[0]]))
                h = o * activations.tanh(c)

                return h, [h, c]

            def step_backward(inputs, states):
                h_tm1 = states[0]  # previous memory state
                c_tm1 = states[1]  # previous carry state

                x_i = tf.tensordot(inputs, self.kernel_i_backward,axes=[[2],[0]])
                x_f = tf.tensordot(inputs, self.kernel_f_backward,axes=[[2],[0]])
                x_c = tf.tensordot(inputs, self.kernel_c_backward,axes=[[2],[0]])
                x_o = tf.tensordot(inputs, self.kernel_o_backward,axes=[[2],[0]])
                x_i = K.bias_add(x_i, self.bias_i_backward)
                x_f = K.bias_add(x_f, self.bias_f_backward)
                x_c = K.bias_add(x_c, self.bias_c_backward)
                x_o = K.bias_add(x_o, self.bias_o_backward)
                i = activations.hard_sigmoid(x_i + tf.tensordot(h_tm1,
                                                          self.recurrent_kernel_i_backward,axes=[[2],[0]]))
                f = activations.hard_sigmoid(x_f + tf.tensordot(h_tm1,
                                                          self.recurrent_kernel_f_backward,axes=[[2],[0]]))
                c = f * c_tm1 + i * activations.tanh(x_c + tf.tensordot(h_tm1,
                                                                self.recurrent_kernel_c_backward,axes=[[2],[0]]))
                o = activations.hard_sigmoid(x_o + tf.tensordot(h_tm1,
                                                          self.recurrent_kernel_o_backward,axes=[[2],[0]]))
                h = o * activations.tanh(c)

                return h, [h, c]

            # Make all-zero initial state. 
            # Directly obtaining the first input dimension is not allowed, so this is the work-aronud.
            initial_state = tf.tensordot(K.max(proj_input*0,axis=2),K.zeros((self.proj_dim, self.rec_hid_dim)), axes = [[2],[0]])
            proj_input_permute = K.permute_dimensions(proj_input,(0,2,1,3))
            proj_input_permute_backward = K.reverse(proj_input_permute, 1)
            _,all_rnn_out_forward,_ = K.rnn(step_forward,proj_input_permute,[initial_state,initial_state])
            _,all_rnn_out_backward,_ = K.rnn(step_backward,proj_input_permute_backward,[initial_state,initial_state])
            all_rnn_out = K.concatenate([all_rnn_out_forward,all_rnn_out_backward],axis=-1)
            raw_att_scores = tf.tensordot(K.permute_dimensions(all_rnn_out,(0,2,1,3)), 
                                                self.att_scorer, axes=[[3],[0]])


        elif self.context == 'para':
            raw_att_scores = K.sum(tf.tensordot(proj_input, self.att_scorer, axes=[[3],[2]]), axis = [1, 2]) # (sample,c,w)
        
        if self.hard: # Hard attention
            rep_att_score = K.repeat_elements(K.expand_dims(raw_att_scores),rep=self.wd,axis=-1)
            top = tf.nn.top_k(K.permute_dimensions(rep_att_score,(0,1,3,2)),k=self.k).indices
            permute_X = K.permute_dimensions(X,(0,1,3,2))
            reduced_X = K.permute_dimensions(tf.batch_gather(permute_X, top),(0,1,3,2))
            new_att_scores = K.softmax(tf.nn.top_k(raw_att_scores,k=self.k).values,axis=2)
            result = K.batch_dot(new_att_scores,reduced_X,axes=[2,2])
        else:
            att_scores = K.softmax(raw_att_scores, axis=2)
            result = K.batch_dot(att_scores,X,axes=[2,2]) # (sample,c,d)
        if self.return_attention:
            return [result, raw_att_scores]
        else:
            return result

    def get_config(self):
        return {'name': 'TensorAttention',
                'att_input_shape': (self.td1, self.td2, self.wd),
                'proj_dim': self.proj_dim,
                'rec_hid_dim': self.rec_hid_dim,
                'hard_k': self.k,
                'context': self.context,
                'trainable': True}

