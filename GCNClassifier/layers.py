# %load layers.py
from inits import *
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res    
    
class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])

class Flatten(Layer):
    """Flattens a tensor layer."""
    def __init__(self, **kwargs):
        super(Flatten, self).__init__(**kwargs)

    def _call(self, inputs):
        x = inputs
        # flatten the tensor to one layer
        shape = x.get_shape().as_list()               # a list: [None,...]
        dim = np.prod(shape[1:])                   # dim = prod(...)
        x_flattened = tf.reshape(x, [-1, dim])        # -1 means "all"
        return x_flattened
    
class Dense(Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.featureless = featureless
        self.bias = bias

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        x = tf.nn.dropout(x, 1-self.dropout)
        
        # add dummy dimension
        x = tf.expand_dims(x, 1) # Batch1M
        
        # transform
        (batch, n, m) = x.get_shape().as_list()
        p = self.vars['weights'].get_shape().as_list()[1]
        output = tf.reshape(tf.reshape(x, [-1, m]) @ self.vars['weights'], [-1, n, p]) # BatchNM * MP => Batch1P (N should be 1)
        
        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output) # Batch1P

class GraphConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 act=tf.nn.relu, bias=False,
                **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.bias = bias
        self.num_nodes = self.support.get_shape().as_list()[2]
        self.output_dim = output_dim
        
        with tf.variable_scope(self.name + '_vars'):
            # make all weight matrices for supports in convolution
            for i in range(self.support.get_shape().as_list()[1]):# support: ?xSupportsxNxNxM
                for j in range(self.support.get_shape().as_list()[4]):
                    tensor_name = 'weights_support_' + str(i) + '_M_' + str(j)
                    self.vars[tensor_name] = glorot([input_dim, output_dim], name=tensor_name)
            # make vector to do weighted sum of all convolved features (w in SUM(wi*(NxF')) for w in M)
            self.vars["Features Combination"] =tf.Variable(tf.random_uniform([self.support.get_shape().as_list()[4]]))
            #uniform(self.support.get_shape().as_list()[4], name="Features Combination")
            # make bias matrice
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')
            
        
        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs
        
        # dropout
        x = tf.nn.dropout(x, 1-self.dropout)
        
        # convolve
        convolved_features = []
        for j in range(self.support.get_shape().as_list()[4]):
            temp = []
            for i in range(self.support.get_shape().as_list()[1]):
                tensor_name = 'weights_support_' + str(i) + '_M_' + str(j)
                # BatchNF * FF' weight tensor (num_nodes, |Features'|)
                (N, F) = x.get_shape().as_list()[1:]
                embed = tf.reshape(x, [-1, F])
                pre_sup =  tf.reshape(tf.reshape(x, [-1, F]) @ self.vars[tensor_name], [-1, N, self.output_dim])
                (batch, _, F_new) = pre_sup.get_shape().as_list()

                # BatchNN * BatchNF' => BatchNF'
                support = tf.slice(self.support, [0,i,0,0,j], [-1,1,-1,-1,1]) # get Batch1NN1
                support = tf.reshape(support, [-1,N,N]) # reshape to BatchNN
                support = support @ pre_sup # now BatchNF'
                temp.append(support)
            # adds together list of BatchNF' into one BatchNF' for a single original adjacency matrix
            convolved_F = tf.add_n(temp)
            convolved_features.append(convolved_F)
        # stack list into one tensor of shape BatchNF'M
        convolved_features = tf.stack(convolved_features, axis = 3)
        # do weighted multiplication
        convolved_features = tf.multiply(convolved_features, self.vars["Features Combination"])
        # sum together to remove 4th dimension
        output = tf.reduce_sum(convolved_features, axis = 3)
        
        # bias
        if self.bias:
            output += self.vars['bias'] # Broadcasting spreads bias across Batch and Node dimensions

        return self.act(output)

class SelfAttention(Layer):
    """Self attention layer, input is in ?xNxhidden, output is in ?x(Bias*Hidden). Hidden should correspond 
    to the number of features nodes have."""
    
    def __init__(self, attention_dim, bias_dim, hidden_units, placeholders, dropout=0., **kwargs):
        super().__init__(**kwargs)
        
        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.
        
        self.hidden_units = hidden_units
        self.A = None
        with tf.variable_scope(self.name + '_vars'):
            self.vars['Ws'] = tf.Variable(tf.random_uniform([attention_dim, self.hidden_units])) # AttentionxHidden
            self.vars['W2'] = tf.Variable(tf.random_uniform([bias_dim, attention_dim])) # BiasxAttention

    def _call(self, inputs):
        
        # dropout
        inputs = tf.nn.dropout(inputs, 1-self.dropout)
        
        # AttentionxHidden * ?xHiddenxN => ?xAttentionxN
        inputsT = tf.transpose(inputs, perm = [0, 2, 1]) # transpose the inner matrices which is our intention
        
        #print("inputsT is of shape {}".format(inputsT.get_shape().as_list()))
        #print("self.vars[Ws] is of shape {}".format(self.vars['Ws'].get_shape().as_list()))
        
        aux = tf.einsum('ah,bhn->ban', self.vars['Ws'], inputsT)
        aux = tf.tanh(aux)
        
        #print("aux is of shape {}".format(aux.get_shape().as_list()))
        
        
        # BiasxAttention * ?xAttentionxN => ?xBiasxN
        self.A = tf.einsum('ba,uan->ubn',self.vars['W2'], aux)
        self.A = tf.nn.softmax(self.A)
        #print("A is of shape {}".format(self.A.get_shape().as_list()))
        #tf.summary.histogram('self_attention', self.A) For visualization
        
        # ?xBiasxN * ?xNxHidden => ?xBiasxHidden
        out = self.A @ inputs
        #print("out is of shape {}".format(out.get_shape().as_list()))
        
        # ?xBiasxHidden => ?x(Bias*Hidden)
        out = tf.reshape(out, [ -1, out.get_shape().as_list()[1] * out.get_shape().as_list()[2]])
        #print("out is of shape {}".format(out.get_shape().as_list()))
        return out
