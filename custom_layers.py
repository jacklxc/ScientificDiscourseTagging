# Simple modification to the implementation of TimeDistributedDense in keras
import tensorflow as tf
from keras.engine.topology import Layer
from keras import backend as K
from keras import activations, initializers, regularizers, constraints

class HigherOrderTimeDistributedDense(Layer):
    '''Apply the same dense layer on all inputs over two time dimensions.
    Useful when the input to the layer is a 4D tensor.

    # Input shape
        4D tensor with shape `(nb_sample, time_dimension1, time_dimension2, input_dim)`.

    # Output shape
        4D tensor with shape `(nb_sample, time_dimension1, time_dimension2, output_dim)`.

    # Arguments
        output_dim: int > 0.
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)),
            or alternatively, Theano function to use for weights
            initialization. This parameter is only relevant
            if you don't pass a `weights` argument.
        activation: name of activation function to use
            (see [activations](../activations.md)),
            or alternatively, elementwise Theano function.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
        weights: list of numpy arrays to set as initial weights.
            The list should have 1 element, of shape `(input_dim, output_dim)`.
        input_dim: dimensionality of the input (integer).
            This argument (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.
    '''
    input_ndim = 4

    def __init__(self, output_dim,
                 init='glorot_uniform', activation='linear', weights=None,
                 input_dim=None, reg=0, **kwargs):
        self.supports_masking = True 
        self.output_dim = output_dim
        self.init = initializers.get(init)
        self.reg = reg
        self.activation = activations.get(activation)
        self.initial_weights = weights

        self.input_dim = input_dim
        super(HigherOrderTimeDistributedDense, self).__init__(**kwargs)

    def build(self,input_shape):
        input_dim = input_shape[3]
        self.W = self.add_weight(name='W',shape=(input_dim, self.output_dim),initializer=self.init, 
                                 regularizer = regularizers.l2(self.reg), trainable=True)
        self.b = self.add_weight(name='b',shape=(1,1,1,self.output_dim,),initializer=self.init, 
                                 regularizer = regularizers.l2(self.reg), trainable=True)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        super(HigherOrderTimeDistributedDense, self).build(input_shape)
        
    def compute_output_shape(self,input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.output_dim)

    def compute_mask(self, input, input_mask=None):   
        return input_mask   

    def call(self, X, mask=None):
        outputs = tf.tensordot(X,self.W,axes=[[3],[0]]) + self.b
        outputs = self.activation(outputs)
        if mask is not None:
            outputs = K.expand_dims(K.cast(mask,"float32"),axis=3) * outputs
        return outputs
    
    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'output_dim': self.output_dim,
                  'init': 'glorot_uniform',
                  'reg': self.reg,
                  'activation': self.activation.__name__,
                  'input_dim': self.input_dim}
        base_config = super(HigherOrderTimeDistributedDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
