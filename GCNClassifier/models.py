# %load models.py
from layers import *
from metrics import *

flags = tf.app.flags
FLAGS = flags.FLAGS

class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None
        self.logits = None
        self.predictions = None
        
        self.loss = 0
        self.accuracy = 0
        self.f1_score = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()
        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class GCN(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN, self).__init__(**kwargs)
        self.inputs = placeholders['features']
        self.input_dim = input_dim
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders
        self.number_nodes = placeholders['features'].get_shape().as_list()[1]
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        
        if FLAGS.balanced_training == "True":
            labels = self.placeholders['labels']
            logits = self.outputs
            # Get relative frequency of each class
            class_counts = tf.reduce_sum(labels, 0)
            class_frequencies = class_counts / tf.reduce_sum(class_counts)
            # Cross entropy error
            entropies = tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = labels, dim = -1)
            # Scale by 1/frequency
            scalers = labels / class_frequencies
            scalers = tf.reduce_sum(scalers, 1)
            entropies_scaled = scalers * entropies
            self.loss += tf.reduce_mean(entropies_scaled)
        else:
            # Cross entropy error
            labels = self.placeholders['labels']
            logits = self.outputs
            entropies = tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = labels, dim = -1)
            loss = tf.reduce_mean(entropies)
            self.loss += loss

    def _accuracy(self):
        labels = self.placeholders['labels']
        logits = self.outputs
        self.logits = logits
        labels=tf.argmax(labels, 1) # labels
        predictions=tf.argmax(logits, 1) # prediction as one hot
        self.predictions = predictions
        
        # Define the metric and update operations, f1 score is also calculated
        tf_metric, tf_metric_update = tf.metrics.accuracy(predictions = predictions, labels = labels, name = "accuracy")
        self.accuracy = tf_metric_update
        
        # Isolate the variables stored behind the scenes by the metric operation
        running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="accuracy")
        
        # Define initializer to initialize/reset running variables
        self.running_vars_initializer = tf.variables_initializer(var_list=running_vars)

    def _build(self):
        # Parse through the graph convolutional layer specifications and the hidden layers
        def parse_array(string):
            sl = list(string)
            if sl[0] != "[" and sl[-1] == "]":
                raise ValueError("Invalid dimensions input")
            string = string.strip("[]")
            string = string.replace(" ", "")
            num_ls = string.split(",")
            return [int(x) for x in num_ls if x != ""]
        graph_convolution_dimensions = parse_array(FLAGS.graph_conv_dimensions)
        fully_connected_dimensions = parse_array(FLAGS.connected_dimensions)
        
        # Graph Convolutional Layers
        prior_dimension = self.input_dim
        for gcdim in graph_convolution_dimensions:
            self.layers.append(GraphConvolution(input_dim=prior_dimension,
                                                output_dim=gcdim,
                                                placeholders=self.placeholders,
                                                act=tf.nn.relu,
                                                dropout=True,
                                                logging=self.logging))
            prior_dimension = gcdim
        
        # Self Attention
        self.layers.append(SelfAttention(attention_dim=FLAGS.attention_dim,
                                         bias_dim=FLAGS.attention_bias, 
                                         hidden_units=prior_dimension,
                                         placeholders=self.placeholders,
                                         dropout=True,
                                         logging=self.logging))
        
        # Fully Connected Layers
        fully_connected_dimensions.append(self.output_dim)
        prior_dimension = prior_dimension * FLAGS.attention_bias
        for fcdim in fully_connected_dimensions:
            self.layers.append(Dense(input_dim=prior_dimension,
                                     output_dim=fcdim,
                                     act=tf.nn.relu,
                                     placeholders=self.placeholders,
                                     dropout=True,
                                     logging=self.logging))
            prior_dimension = fcdim
        
    def predict(self):
        logits = self.outputs
        return tf.nn.softmax(logits)
