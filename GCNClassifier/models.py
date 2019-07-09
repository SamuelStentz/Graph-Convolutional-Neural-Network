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
        #print("\n\n\nactivations\n{}\n\n\n".format(self.activations))
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
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders
        self.number_nodes = placeholders['features'].get_shape().as_list()[1]
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        
        #print("placeholders and descriptions:")
        #for key in self.placeholders:
        #    print("{}: {}".format(key, self.placeholders[key]))
        
        #print("Model's input {}, output {}".format(self.input_dim, self.output_dim))
        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        
        # Cross entropy error
        labels = self.placeholders['labels']
        logits = self.outputs
        dim = logits.get_shape().as_list()[2]
        logits = tf.reshape(logits, [-1, dim])
        entropies = tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = labels, dim = -1)
        loss = tf.reduce_mean(entropies)
        #print("Labels {} \nLogits {}\nEntropies {}\nLoss {}\n\n".format(labels, logits, entropies, loss))
        self.loss += loss

    def _accuracy(self):
        labels = self.placeholders['labels']
        logits = self.outputs
        dim = logits.get_shape().as_list()[2]
        logits = tf.reshape(logits, [-1, dim])
        self.logits = logits
        labels=tf.argmax(labels, 1) # labels
        predictions=tf.argmax(logits, 1) # prediction as one hot
        self.predictions = predictions
        
        # Define the metric and update operations
        tf_metric, tf_metric_update = tf.metrics.accuracy(predictions = predictions, labels = labels, name = "accuracy")
        self.accuracy = tf_metric_update
        
        # Isolate the variables stored behind the scenes by the metric operation
        running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="accuracy")
        # Define initializer to initialize/reset running variables
        self.running_vars_initializer = tf.variables_initializer(var_list=running_vars)

    def _build(self):

        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            logging=self.logging))
        
        #print("first layer: input {}, output {}".format(self.input_dim, FLAGS.hidden1))
        
        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=FLAGS.hidden2,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            logging=self.logging))
    
        #print("second layer: input {}, output {}".format(FLAGS.hidden1, FLAGS.hidden2))
        
        ####### flatten the output so that it is now one continuous layer
        #####self.layers.append(Flatten(logging=self.logging))
        
        # perform self attention, hidden must be number of features, or hidden2. Other two are more arbitrary
        self.layers.append(SelfAttention(attention_dim = FLAGS.attention_dim,
                                         bias_dim = FLAGS.attention_bias, 
                                         hidden_units = FLAGS.hidden2,
                                         placeholders=self.placeholders,
                                         dropout=True,
                                         logging = self.logging))
        
        #print("self attention with attention_dim {}, bias_dim {}:\n input {}, output(bias*input) {}".format(FLAGS.attention_dim, FLAGS.attention_bias, [self.number_nodes, FLAGS.hidden2] ,FLAGS.hidden2 * FLAGS.attention_bias))
        
        # dense FC layer to get out an output prediction
        self.layers.append(Dense(input_dim=FLAGS.hidden2 * FLAGS.attention_bias,
                                 output_dim=self.output_dim,
                                 act=lambda x: x,
                                 placeholders=self.placeholders,
                                 dropout=True,
                                 logging=self.logging))
        
        #print("final layer: input {}, output {}".format(FLAGS.hidden2 * FLAGS.attention_bias, self.output_dim))
        
    def predict(self):
        logits = self.outputs
        dim = logits.get_shape().as_list()[2]
        logits = tf.reshape(logits, [-1, dim])
        return tf.nn.softmax(logits)
